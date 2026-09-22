"""[E10 2026-09-12] Re-stage E2's ``required.json`` per RU from the VERIFIER's own required-layout function.

Why: E1's producer writes every computed gate and its ``required_gates.bin`` was inventoried under the v1 rows (4,622,106 words for
request ``2663944e...``).  Under E6's ``a9/legacy-minus-inputs/v2`` population the verifier's live ``required=`` source
(``verifier_v3.steps.required_source_live`` -> E3 ``PositionOracle`` -> E2 ``InteriorLayout.build``) names 3,069,306 words (no scan words of
the regrouped ``TokenSelect_v1|whole-scan`` unit, no compiled literals).  A prover tree framed over the v1 list is refused at step 4
(``diag3`` 00:14Z: ``INTERIOR_WIDTH_MISMATCH`` ``layers.0/pre`` 3590 vs 3530 words, ``INTERIOR_REGION_MISSING``).

What: for a record whose commitments exist (phase 0/1 done -- the boundary id lists are what E3's oracle needs; they do not depend on the
interior tree), call the shipped ``required_source_live`` exactly as the driver does and write, per staged RU directory
``<values>/<req>/<step>/``, ``required.json`` = ``{"gates_bin": "required_v2_gates.bin", "dtype": "<i8", "widths": [...]}`` in E2's
``_read_required`` form; E1's original list is kept as ``required.e1.json`` (+ its ``required_gates.bin``).  The prover then commits exactly the
verifier's layout.  The served boundary of this request is deterministic across attempts (served bits identical to attempt 6 / prep / diag1-3),
so the record used to derive the set may be an earlier attempt; the provenance (record path, tree sha, population digest) is written into
``required.json#source`` and the staging marker.

Usage (pod):  $PY -m record_v5.interior_required_v2 --record <REC with adapter/selective/commitments.json> --values <STAGED> [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def prover_required(layout: Any) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The PROVER's framing of one RU's required set from the verifier's layout object: ``(gates in layout order, widths per word, leaf ids)``.

    [E10 D40 T-3, 2026-09-12] Since D38 (wave 4, 5f4d888e) ``required_source_live`` returns E2's ``LazyInteriorLayout`` (O(largest call) memory,
    no ``_all_gates``); reading the private field crashed here, ``run_tiny1_pod.sh`` went on with E1's v1 inventory list and E8's 32x16 n2 was
    refused at step 4 ``INTERIOR_WIDTH_MISMATCH (3590 != 3530 words at t=0/layers.0/pre/w=32)``.  The prover holds every word anyway: E2's PUBLIC
    ``materialize() -> InteriorLayout`` (E2 report 08:45Z: same ids / ranks / positions) gives the eager layout, whose PUBLIC ``gates`` property
    (layout order) and ``leaves`` frame the set.  ``tests/contract/e10_interior_required_v2_test.py`` asserts this equals the verifier's lazy
    layout by identity and count on the 32x15 instance."""
    if hasattr(layout, "materialize") and not hasattr(layout, "gates"):
        layout = layout.materialize()
    gates = np.asarray(layout.gates, dtype=np.int64)
    leaves = list(layout.leaves)
    widths = np.concatenate([np.full(lf.words, int(lf.width), dtype=np.int64) for lf in leaves]) if leaves else np.zeros(0, dtype=np.int64)
    if widths.shape[0] != gates.shape[0]:
        raise ValueError(f"prover_required: {gates.shape[0]} gates but {widths.shape[0]} leaf words")
    return gates, widths, [str(lf.canonical_id) for lf in leaves]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--record", required=True, help="a record dir with protocol_instance.json, population.v5.json, adapter/selective/commitments.json")
    ap.add_argument("--values", required=True, help="the staged --interior-values dir (<values>/<req>/<step>/)")
    ap.add_argument("--dry-run", action="store_true", help="derive and report, write nothing")
    a = ap.parse_args(argv)
    rec = Path(a.record)
    values = Path(a.values)
    inst_doc = _load(rec / "protocol_instance.json")
    pop_doc = _load(rec / "population.v5.json")
    sel = rec / "adapter" / "selective"
    c2 = _load(sel / "commitments.json")
    request_ids = list((inst_doc.get("epoch") or {}).get("request_ids") or [])
    staged_reqs = sorted(p.name for p in values.iterdir() if p.is_dir() and not p.name.startswith("."))
    retained = [r for r in request_ids if r in staged_reqs]
    if not retained:
        print(f"[interior_required_v2] no staged request of the instance under {values}: instance {request_ids}, staged {staged_reqs}")
        return 2

    from e2e.verifier_v3 import steps  # the shipped verifier's function, not a copy

    t0 = time.perf_counter()
    src, label, facts = steps.required_source_live(inst_doc, pop_doc, sel, c2, retained)
    if src is None:
        print(f"[interior_required_v2] REFUSED: the verifier's required source did not build: {label} {facts}")
        return 3
    freeze = _load(rec / "freeze.json") if (rec / "freeze.json").exists() else {}
    tree_sha = str(freeze.get("commit") or freeze.get("sha") or "")[:12]
    pdig = str((pop_doc.get("partition") or {}).get("partition_digest") or pop_doc.get("partition_digest") or "")[:16]
    provenance = (f"verifier_v3.steps.required_source_live [{label}] over record {rec.name} (tree {tree_sha or '?'}), population {pop_doc.get('schema')} "
                  f"partition {pdig}, instance {(inst_doc.get('epoch') or {}).get('epoch_id')}; replaces E1's v1-inventory list (kept as required.e1.json)")
    total_v2 = total_e1 = 0
    summary: dict[str, dict] = {}
    for req in retained:
        for step_dir in sorted(p for p in (values / req).iterdir() if p.is_dir()):
            layout = src(req, step_dir.name)
            if layout is None:
                print(f"[interior_required_v2] {req}/{step_dir.name}: the verifier names NO interior region for this RU (not a retained interior RU) -- left as is")
                continue
            gates, widths, _ids = prover_required(layout)  # [D40 T-3] lazy or eager layout -> the prover's framing (see prover_required)
            n_leaves = len(_ids)
            e1_json = step_dir / "required.json"
            e1_count = None
            if e1_json.exists():
                try:
                    e1_doc = _load(e1_json)
                    e1_count = int(e1_doc.get("count")) if isinstance(e1_doc, dict) and e1_doc.get("count") is not None else None
                    if isinstance(e1_doc, dict) and "gates_bin" in e1_doc and e1_count is None:
                        e1_count = int(np.fromfile(step_dir / e1_doc["gates_bin"], dtype=e1_doc.get("dtype", "<u4")).shape[0])
                except (OSError, ValueError, TypeError):
                    e1_count = None
            summary[f"{req}/{step_dir.name}"] = {"v2_words": int(gates.shape[0]), "leaves": n_leaves, "e1_words": e1_count}
            total_v2 += int(gates.shape[0])
            total_e1 += int(e1_count or 0)
            if a.dry_run:
                continue
            if e1_json.exists() and not (step_dir / "required.e1.json").exists():
                e1_json.rename(step_dir / "required.e1.json")
            gates.astype("<i8").tofile(step_dir / "required_v2_gates.bin")
            (step_dir / "required.json").write_text(json.dumps({
                "gates_bin": "required_v2_gates.bin", "dtype": "<i8", "count": int(gates.shape[0]), "widths": widths.tolist(),
                "leaves": n_leaves, "rule": layout.rule, "chunk_words": layout.chunk_words, "tree": layout.tree,
                "source": provenance, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }) + "\n")
    dt = time.perf_counter() - t0
    print(f"[interior_required_v2] {'DRY-RUN ' if a.dry_run else ''}{len(summary)} RU(s): verifier-required {total_v2:,} words (E1's v1 list {total_e1:,}) in {dt:.1f}s; {label}")
    for k, v in summary.items():
        print(f"  {k}: v2 {v['v2_words']:,} words / {v['leaves']} leaves; E1 list {v['e1_words']}")
    if not a.dry_run:
        (values / ".required-v2.json").write_text(json.dumps({"source": provenance, "totals": {"v2_words": total_v2, "e1_words": total_e1}, "per_ru": summary,
                                                             "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
