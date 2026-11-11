"""`checks/coverage.py` recomputes `coverage_check` from ONE side's files: the candidate's manifest, instrumented layouts and runs when a
candidate names the row, else the reference's.  #73's first side run checked the candidate's manifest against the frozen reference's
layouts and found its norm_scales family (69,020 identities, committed by the norm-scale source into the candidate's own layouts)
missing entirely, while the Commit's own check said missing 0."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from tests.regression import resolver
from tests.regression.checks import coverage

LAYOUTS = "commit/layouts_pair0_instrumented.json.gz"

NORMS = [f"model.layers.{i}.input_layernorm" for i in range(3)]
OUTS = [f"model.layers.{i}.mlp" for i in range(3)]


def _manifest() -> dict:
    """A stand-in norm-scale manifest: each layer's MLP output and its norm's scale, over two steps."""
    ids = []
    for step in (0, 1):
        ids += [{"family": "instance_outputs", "request_id": "r0", "rank": 0, "step": step, "op_path": op, "output_member": "out",
                 "invocation": 0, "element_range": [0, 8]} for op in OUTS]
        ids += [{"family": "norm_scales", "request_id": "r0", "rank": 0, "step": step, "op_path": op, "output_member": "norm_scale",
                 "invocation": 0, "element_range": [0, 1]} for op in NORMS]
    return {"complete": True, "manifest_digest": "m" * 64, "identities": ids}


def _layouts(*, norms: bool = True, drop: str | None = None) -> dict:
    """The instrumented layouts: the MLP outputs, and the norm-scale source's entries unless `norms` is False; `drop` omits one entry."""
    out = {}
    for step in (0, 1):
        metas = [{"name": op, "numel": 8} for op in OUTS]
        metas += [{"name": f"{op}/norm_scale", "numel": 1} for op in NORMS] if norms else []
        out[str(step)] = [m for m in metas if not (step == 1 and m["name"] == drop)]
    return out


def _tree(root: Path, *, manifest: bool = True, layouts: dict | None = None) -> Path:
    (root / "commit").mkdir(parents=True)
    if manifest:
        (root / "manifest.json").write_text(json.dumps(_manifest()))
    if layouts is not None:
        with gzip.open(root / LAYOUTS, "wt") as f:
            json.dump(layouts, f)
    (root / "commit" / "runs.jsonl").write_text(json.dumps({"arm": "instrumented", "committer": "native_collect",
                                                            "commit": {"binding": {"coverage": {"ok": True, "checked": 12, "missing_n": 0}}}}) + "\n")
    return root


class _Row:
    """The two sides a check reads: the reference's files (`path` / `has`) and, when given, the candidate's (`candidate`)."""

    key = "stand-in-norm-scale-row"

    def __init__(self, reference: Path, candidate: Path | None = None) -> None:
        self.reference, self.cand = reference, candidate

    def candidate_root(self):
        return self.cand

    def candidate(self, rel):
        return (self.cand / rel) if self.cand is not None and (self.cand / rel).is_file() else None

    def has(self, rel):
        return (self.reference / rel).is_file()

    def path(self, rel, verify=True):
        if not self.has(rel):
            raise resolver.NotResolvable(rel)
        return self.reference / rel

    def jsonl(self, rel):
        return [json.loads(line) for line in open(self.path(rel)) if line.strip()]


def test_the_candidates_norm_scales_are_covered_by_the_candidates_own_layouts(tmp_path):
    """#73's case: the reference run committed no norm scales; the candidate's source did, into its own layouts."""
    row = _Row(_tree(tmp_path / "reference", layouts=_layouts(norms=False)), _tree(tmp_path / "candidate", layouts=_layouts()))
    assert coverage.applies(row) is None
    act = coverage.actual(row)
    assert act["ok"] is True and act["checked"] == 12 and act["missing_n"] == 0 and act["families_missing_entirely"] == [], act
    assert act["source"] == "recomputed from the candidate" and act["recorded"] is True
    stale = coverage.actual(_Row(row.reference))                          # the reference checked against its own layouts
    assert stale["source"] == "recomputed" and stale["families_missing_entirely"] == ["norm_scales"] and stale["missing_n"] == 6


def test_a_norm_scale_entry_missing_from_the_candidates_layouts_is_missing(tmp_path):
    row = _Row(_tree(tmp_path / "reference", layouts=_layouts()), _tree(tmp_path / "candidate", layouts=_layouts(drop=f"{NORMS[1]}/norm_scale")))
    act = coverage.actual(row)
    assert act["ok"] is False and act["missing_n"] == 1 and act["missing_by_family"] == {"norm_scales": 1} and act["families_missing_entirely"] == []


def test_a_candidate_without_its_own_layouts_is_not_checked_against_the_references(tmp_path):
    row = _Row(_tree(tmp_path / "reference", layouts=_layouts()), _tree(tmp_path / "candidate", layouts=None))
    assert coverage.applies(row) == "no instrumented layouts on the candidate (Commit never ran)"
    with pytest.raises(resolver.NotResolvable, match="the candidate lacks commit/layouts_pair0_instrumented.json.gz"):
        coverage.actual(row)
    row = _Row(_tree(tmp_path / "reference2", layouts=_layouts()), _tree(tmp_path / "candidate2", manifest=False, layouts=_layouts()))
    assert coverage.applies(row).startswith("manifest.json not resolvable on the candidate")


def test_without_a_candidate_the_reference_reproduces_itself(tmp_path):
    row = _Row(_tree(tmp_path / "reference", layouts=_layouts()))
    act, exp = coverage.actual(row), coverage.expected(row)
    assert act["ok"] is True and act["missing_n"] == 0 and act["source"] == "recomputed" and act["recorded"] is True
    assert exp == {"ok": True, "checked": 12, "missing_n": 0, "recorded": True}
