"""(e) Identity coverage reproduces: `manifest_format.coverage_check(manifest, layouts)` with the CURRENT code over the recorded
manifest of record and the instrumented run's recorded layouts (`commit/layouts_pair<k>_instrumented.json.gz`: {step: [{name, numel|
shape}]}) must state what the Commit's binding recorded (`commit/runs.jsonl`, treatment entries: `commit.binding.coverage` --
ok / checked / missing_n / committed_extras_n / duplicate_names_n / manifest_digest).

Inputs: `manifest.json` (39-222 MB: pod, or the fetched manifest artifact) + the layouts (small).  STRONG on the counted facts
(a producer omitted from the layouts is a missing identity by family); the binding map itself (`commit/binding_map_p*.json`, 0.5-1.5
GB each) is NOT read here -- value correspondence is bind's separate check and is listed as a gap.

The manifest, the layouts and the recorded coverage are read from ONE side: the candidate's when a candidate names the row (all three, or
NotResolvable naming the file it lacks), else the reference's.  A candidate manifest checked against the reference's layouts reports
every family the reference run did not commit as missing entirely -- #73's norm_scales (69,020 identities), committed by the norm-scale
source into the candidate's own layouts.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from .. import resolver

NAME = "coverage"
TIER = "T0"
STRENGTH = "strong"
INPUTS = ("manifest.json", "commit/layouts_pair0_instrumented.json.gz", "commit/runs.jsonl")
SCALARS = ("ok", "checked", "missing_n", "committed_unmatched_n", "committed_extras_n", "duplicate_names_n", "manifest_digest", "error")


LAYOUTS = "commit/layouts_pair0_instrumented.json.gz"


def _side(row: resolver.Row, rel: str) -> Path | None:
    """`rel` on the side `actual` reads: the candidate's when a candidate names the row, else the reference's (None: not there)."""
    if row.candidate_root() is not None:
        return row.candidate(rel)
    return row.path(rel) if row.has(rel) else None


def applies(row: resolver.Row) -> str | None:
    side = "the candidate" if row.candidate_root() is not None else "this row"
    if _side(row, LAYOUTS) is None:
        return f"no instrumented layouts on {side} (Commit never ran)"
    if _side(row, "manifest.json") is None:
        return f"manifest.json not resolvable on {side} (pod / fetched manifest artifact needed)"
    return None


def _recorded_coverage(runs: Path | None) -> dict | None:
    if runs is None:
        return None
    for r in (json.loads(line) for line in open(runs) if line.strip()):
        if r.get("arm") == "control" or r.get("committer") in (None, "control"):
            continue
        b = (r.get("commit") or {}).get("binding") if isinstance(r.get("commit"), dict) else None
        cov = (b or {}).get("coverage") if isinstance(b, dict) else None
        if isinstance(cov, dict):
            return cov
    return None


def _scalars(cov: dict) -> dict:
    return {k: cov.get(k) for k in SCALARS if k in cov}


def expected(row: resolver.Row) -> dict:
    cov = _recorded_coverage(row.path("commit/runs.jsonl") if row.has("commit/runs.jsonl") else None)
    if cov is None:
        return {"ok": True, "recorded": False}
    out = _scalars(cov)
    out["recorded"] = True
    if isinstance(cov.get("missing_by_family"), dict):
        out["missing_by_family"] = cov["missing_by_family"]
    return out


def actual(row: resolver.Row) -> dict:
    from verity_vllm.query.manifest.format import coverage_check
    cand = row.candidate_root() is not None
    man_path, lay_path = _side(row, "manifest.json"), _side(row, LAYOUTS)
    if man_path is None or lay_path is None:
        lacks = [rel for rel, p in (("manifest.json", man_path), (LAYOUTS, lay_path)) if p is None]
        raise resolver.NotResolvable(f"{row.key}: the {'candidate' if cand else 'reference'} lacks {', '.join(lacks)}: its coverage is not "
                                     "recomputed from the other side's files")
    man = json.load(open(man_path))
    with gzip.open(lay_path, "rt") as f:
        lay = json.load(f)
    res = coverage_check(man, lay)
    out = _scalars(res)
    out["recorded"] = _recorded_coverage(_side(row, "commit/runs.jsonl")) is not None
    out["missing_by_family"] = res.get("missing_by_family") or {}
    out["families_missing_entirely"] = res.get("families_missing_entirely") or []
    out["source"] = "recomputed from the candidate" if cand else "recomputed"
    return out
