"""(e) Identity coverage reproduces: `required_manifest.coverage_check(manifest, layouts)` with the CURRENT code over the recorded
manifest of record and the instrumented run's recorded layouts (`commit/layouts_pair<k>_instrumented.json.gz`: {step: [{name, numel|
shape}]}) must state what the Commit's binding recorded (`commit/runs.jsonl`, treatment entries: `commit.binding.coverage` --
ok / checked / missing_n / committed_extras_n / duplicate_names_n / manifest_digest).

Inputs: `manifest.json` (39-222 MB: pod, or the fetched manifest artifact) + the layouts (small).  STRONG on the counted facts
(a producer omitted from the layouts is a missing identity by family); the binding map itself (`commit/binding_map_p*.json`, 0.5-1.5
GB each) is NOT read here -- value correspondence is bind's separate check and is listed as a gap.
"""
from __future__ import annotations

import gzip
import json

from .. import resolver

NAME = "coverage"
TIER = "T0"
STRENGTH = "strong"
INPUTS = ("manifest.json", "commit/layouts_pair0_instrumented.json.gz", "commit/runs.jsonl")
SCALARS = ("ok", "checked", "missing_n", "committed_unmatched_n", "committed_extras_n", "duplicate_names_n", "manifest_digest", "error")


def applies(row: resolver.Row) -> str | None:
    if not row.has("commit/layouts_pair0_instrumented.json.gz"):
        return "no instrumented layouts on this row (Commit never ran)"
    if not row.has("manifest.json"):
        return "manifest.json not resolvable here (pod / fetched manifest artifact needed)"
    return None


def _recorded_coverage(row: resolver.Row) -> dict | None:
    if not row.has("commit/runs.jsonl"):
        return None
    for r in row.jsonl("commit/runs.jsonl"):
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
    cov = _recorded_coverage(row)
    if cov is None:
        return {"ok": True, "recorded": False}
    out = _scalars(cov)
    out["recorded"] = True
    if isinstance(cov.get("missing_by_family"), dict):
        out["missing_by_family"] = cov["missing_by_family"]
    return out


def actual(row: resolver.Row) -> dict:
    from verity_capture.commit.required_manifest import coverage_check
    man_path = row.candidate("manifest.json") or row.path("manifest.json")
    man = json.load(open(man_path))
    with gzip.open(row.path("commit/layouts_pair0_instrumented.json.gz"), "rt") as f:
        lay = json.load(f)
    res = coverage_check(man, lay)
    out = _scalars(res)
    out["recorded"] = _recorded_coverage(row) is not None
    out["missing_by_family"] = res.get("missing_by_family") or {}
    out["families_missing_entirely"] = res.get("families_missing_entirely") or []
    out["source"] = "recomputed"
    return out
