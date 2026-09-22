"""The U record: PAD-BOUND's union-over-termination wrapper as a pipeline stage.

Two parts, both CPU-cheap and deterministic:

1. **Fixture re-check** — every committed family under
   ``fixtures/verity-ir/padded-bound/v0`` is re-evaluated from scratch
   (:func:`veritor.plans.padded_fixtures.check_fixture`): exact enumeration,
   frozen fold on every prefix, wrapper, soundness chain.  A count/bound/bucket
   mismatch or ANY ``bound < exact`` raises — a stop-the-wave signal.
2. **The U record** — ``veritor/u-record/v1``: for each fixture family, the
   committed union/coarse/direct rows (already validated in part 1) are
   summarized, and the family-capacity anchors ``W`` are emitted for the toy
   (``V = 2``) and for the TINY-PAD Serve@3 worked example
   (``D_MAX = 8, V = 64`` — serve3-spec.md §14).

Scope note (GUIDANCE §30 item 9): the union wrapper and the prefix-``W``
formula are **archived research under "prefix-decoded scope"** — the mainline
output model is the designated output tuple with ``W_out`` (the ``tiny``
stage's record carries those anchors).  This stage remains in the pipeline as
regression evidence for the committed fixtures under their own semantics; the
record says so explicitly rather than silently reinterpreting.

Usage: ``python -m e2e.u_record --out <dir>`` from ``vllm-poc/`` with the repo
root on ``PYTHONPATH`` (the e2e_b0.sh ``padded`` stage does this).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


def capacity_rows() -> list[dict[str, object]]:
    from veritor.plans.padded import family_capacity

    rows = []
    for label, d_max, vocab in (
        ("padded toy (fixtures)", 3, 2),
        ("TINY-PAD Serve@3 worked example (serve3-spec.md \u00a714)", 8, 64),
    ):
        rows.append({
            "label": label,
            "d_max": d_max,
            "vocab": vocab,
            "W_incl_empty_bits": round(family_capacity(d_max, vocab, include_empty=True), 9),
            "W_excl_empty_bits": round(family_capacity(d_max, vocab, include_empty=False), 9),
            "basis": "pinned W = log2(sum_{n} V^n) (padded-bound.md \u00a71); exact integer sum",
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output directory for u_record.json")
    ap.add_argument("--fixtures", default=None, help="fixture dir (default: repo fixtures/verity-ir/padded-bound/v0)")
    args = ap.parse_args()

    import veritor
    from veritor.plans.padded_fixtures import FIXTURE_DIR, check_fixture, read_fixtures

    # FIXTURE_DIR is repo-root-relative; anchor it to the editable install's root so the
    # stage works from vllm-poc/ (where e2e_b0.sh runs)
    repo_root = Path(veritor.__file__).resolve().parents[2]
    fdir = Path(args.fixtures) if args.fixtures else (FIXTURE_DIR if FIXTURE_DIR.is_absolute() else repo_root / FIXTURE_DIR)
    docs = read_fixtures(fdir)
    if not docs:
        print(f"u_record: no fixtures under {fdir}", file=sys.stderr)
        return 2

    families = []
    for doc in docs:
        check_fixture(doc)  # raises on any drift or soundness violation
        n_rows = len(doc["rows"])
        worst = max(float(r["slack"]["union_vs_interpreted"]) for r in doc["rows"])
        families.append({
            "name": doc["name"],
            "rows": n_rows,
            "checks_per_row": [len(r["checks"]) for r in doc["rows"]],
            "union_bits": [r["bound"]["union"] for r in doc["rows"]],
            "coarse_bits": [r["bound"]["coarse"] for r in doc["rows"]],
            "direct_bits": [r["bound"]["direct"] for r in doc["rows"]],
            "interpreted_bits": [r["exact"]["interpreted_bits"] for r in doc["rows"]],
            "max_union_slack_bits": round(worst, 9),
            "status": "reproduced",
        })
        print(f"  {doc['name']}: {n_rows} rows re-derived, all soundness checks hold "
              f"(max union slack {worst:.3f} bits)")

    record = {
        "schema": "veritor/u-record/v1",
        "generated": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "scope": "prefix-decoded (ARCHIVED RESEARCH per GUIDANCE §30 item 9: the mainline output model is "
                 "the designated output tuple with W_out; the union wrapper and the prefix-W formula are "
                 "reclassified, not silently reinterpreted -- these rows remain valid regression evidence "
                 "for the committed fixtures under their own prefix-decoded semantics)",
        "mainline_pointer": "W_out anchors for TINY Serve@3 live in the tiny stage's tiny_record.json "
                            "(veritor/tiny-serve3-record/v1, GUIDANCE §30 items 2/3/10; D1 open)",
        "source": {
            "module": "veritor.plans.padded (PAD-BOUND union-over-termination wrapper)",
            "fixtures": str(fdir),
            "frozen_fold": "veritor.plans.bound.bound_of, untouched (wrapper conditions only on the committed staged length)",
        },
        "families": families,
        "soundness": {
            "families_checked": len(families),
            "total_rows": sum(f["rows"] for f in families),
            "violations": 0,
            "meaning": "every committed row re-derived from scratch; union/coarse/direct each >= the exact "
                       "interpreted family on every row; ANY bound < exact would have raised",
        },
        "capacity_anchors": capacity_rows(),
        "pending": "superseded by GUIDANCE §30: the TINY Serve@3 target certifies U for the designated "
                   "output tuple (see the tiny stage), not a union over prefix buckets",
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "u_record.json"
    path.write_text(json.dumps(record, indent=1) + "\n")
    total = record["soundness"]["total_rows"]
    tiny_w = record["capacity_anchors"][1]["W_incl_empty_bits"]
    print(f"u_record: {len(families)} families / {total} rows reproduced, 0 violations; "
          f"TINY-PAD W = {tiny_w} bits -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
