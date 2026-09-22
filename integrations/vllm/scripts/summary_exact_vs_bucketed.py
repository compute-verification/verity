#!/usr/bin/env python3
"""[SUMMARY-C C3] Exact vs bucketed populations from the *same* symbolic builder, side by side.

``veritor.frontier.width.width_table`` builds a KindTable for a padded ``Serve@2`` horizon either exactly
(``bucket_ratio=None``: one attention body per cache length) or bucketed (cache lengths rounded up on a
geometric grid, the table COMPACT/COST-D use at 2^18).  Both are turned into ``Summary`` objects flagged
``exact`` / ``upper`` and compared with :func:`veritor.plans.summary.check_upper_dominates`:

* totals (``N_V``, ``N_R``, ``M_t`` per threshold) must dominate -- a violation is an error;
* per-kind counts (matched on structure: label, size, in/out counts, out bits -- the builders' names carry a row
  ordinal) are reported; violations are listed, not fatal (a rounded-up attention body has a different
  coordinate-level kind mix, so per-kind dominance is not what the bucketing argument promises);
* ``sampler_populations()`` on the ``upper`` summary must raise ``BucketedPopulationError``.

    python scripts/summary_exact_vs_bucketed.py --lp 4 --dmax 8 --out docs/data/summary-c/exact-vs-bucketed-tiny-lp4-d8.json

Every number is MEASURED on the machine named in the record; the workload is a padded proxy (``PROXY_LABEL``).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from summary_probe import TINY_CONFIG  # noqa: E402

PLANS = (("coordinate", "whole"), ("coordinate", "operator"), ("operator", "whole"), ("operator", "operator"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", default="TINY", help="TINY, or a verity_vllm.registry.b1 *_CONFIG name")
    ap.add_argument("--nl", type=int, default=0)
    ap.add_argument("--lp", type=int, default=4)
    ap.add_argument("--dmax", type=int, default=8)
    ap.add_argument("--plans", default=",".join(f"{v}/cu={c}" for v, c in PLANS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from compact_whole_b0 import build_config, maxrss_mb
    from veritor.frontier.padded import PaddedWorkload
    from veritor.frontier.structure import VClassifier
    from veritor.frontier.width import derived_structure, width_table
    from veritor.plans.summary import (
        BucketedPopulationError,
        SummaryMismatch,
        check_upper_dominates,
        summary_from_kind_table,
    )

    if args.case == "TINY":
        config = dict(TINY_CONFIG)
        if args.nl:
            config["NL"] = args.nl
    else:
        config, _ = build_config(args.case, args.nl, [])
    w = PaddedWorkload("B0", config, args.lp, args.dmax)
    if w.feasible:
        raise SystemExit(w.feasible)
    cl = VClassifier()
    t0 = time.perf_counter()
    st = derived_structure(w, ("coordinate", "operator"), cl)
    doc: dict[str, Any] = {
        "schema": "veritor/summary-c-exact-vs-bucketed/v1",
        "label": "MEASURED (laptop); workload is the padded Serve@2 proxy",
        "host": platform.node(),
        "workload": {"case": args.case, "config": {k: v for k, v in config.items() if k != "profile"}, "LP": w.LP, "D_MAX": w.D_MAX, "gates": st.gates},
        "structure_s": round(time.perf_counter() - t0, 3),
        "plans": {},
    }
    ok = True
    for spec in args.plans.split(","):
        vu, cu = spec.split("/cu=")
        t1 = time.perf_counter()
        exact_t = width_table(w, vu, cu, st.gates, bucket_ratio=None, classifier=cl)
        upper_t = width_table(w, vu, cu, st.gates, classifier=cl)
        prog = {"name": w.name, "digest": None, "digest_kind": "padded-proxy"}
        ex = summary_from_kind_table(exact_t.table, program=prog, plan={"name": exact_t.plan, "digest": None, "digest_kind": "padded-proxy"}, exactness="exact")
        up = summary_from_kind_table(upper_t.table, program=prog, plan={"name": upper_t.plan, "digest": None, "digest_kind": "padded-proxy"}, exactness="upper")
        row: dict[str, Any] = {"rows": {"exact": len(exact_t.table.rows), "upper": len(upper_t.table.rows)}, "seconds": None}
        try:
            rep = check_upper_dominates(ex, up, by="structure", strict_kinds=False)
            row["totals"] = "PASS"
            row["N_V"] = rep["N_V"]
            row["N_R"] = rep["N_R"]
            row["M_t"] = {str(k): v for k, v in rep["M_t"].items()}
            row["kind_violations"] = rep["kind_violations"]
            row["shared_kinds"] = len(rep["vu_kinds"])
            row["kinds_only_exact"] = len(rep["kinds_only_exact"])
            row["kinds_only_upper"] = len(rep["kinds_only_upper"])
            row["units_only"] = {"exact": rep["units_only_exact"], "upper": rep["units_only_upper"]}
            row["per_kind_violations"] = {n: rep["vu_kinds"][n] for n in rep["kind_violations"]}
            row["per_kind"] = "PASS" if not rep["kind_violations"] else f"FAIL ({len(rep['kind_violations'])} kinds)"
        except SummaryMismatch as e:
            ok = False
            row["totals"] = f"FAIL: {e}"
        try:
            up.sampler_populations()
            row["sampler_guard"] = "FAIL: upper summary accepted as sampler population"
            ok = False
        except BucketedPopulationError:
            row["sampler_guard"] = "PASS (BucketedPopulationError)"
        row["seconds"] = round(time.perf_counter() - t1, 3)
        doc["plans"][spec] = row
        print(spec, "totals", row["totals"], "per_kind", row.get("per_kind"), "sampler", row["sampler_guard"], flush=True)
    doc["maxrss_mb"] = maxrss_mb()
    doc["status"] = "ok" if ok else "violation"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"wrote {out}  status {doc['status']}  maxrss {doc['maxrss_mb']} MB")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
