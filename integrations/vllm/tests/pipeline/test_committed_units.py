from __future__ import annotations

import json
from types import SimpleNamespace

from verity_vllm.pipeline import committed_units as C
from verity_vllm.pipeline import row_records as R


def _record():
    return C.record([{
        "input": "committed_0", "op": "aten.sin.default", "module": "block", "reason": "test gap",
        "list_entry": {"op": "aten.sin.default", "module": "block", "reason": "test gap"},
    }])


def test_build_summary_carries_units_and_manifest_guard_refuses_them(tmp_path):
    for wrapper in ("step", "request"):
        out = tmp_path / f"build_{wrapper}"
        out.mkdir()
        (out / "result.json").write_text(json.dumps({
            "ok": True, "program": {"digest": wrapper * 8}, "committed_units": _record(),
        }))
    assert R.build_summary(str(tmp_path)) == "PASS"
    summary = json.loads((tmp_path / "build_summary.json").read_text())
    assert summary["request"]["committed_units"]["inputs"] == ["committed_0"]

    manifest = {"query": {}, "program_digest": "digest", "committed_units": _record()}
    slug, reason = R.manifest_not_of_record(manifest, "digest", SimpleNamespace())
    assert slug == "unverified-committed-values"
    assert "approval has not been recorded" in reason
