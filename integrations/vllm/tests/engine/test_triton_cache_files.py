"""`code_identity._triton_cache_files` finds a compiled kernel's cache entry by the `hash` its metadata json carries, and skips a
json it can't read (not JSON, not UTF-8) instead of failing the lookup.
"""
from __future__ import annotations

import json

from verity_vllm.engine import code_identity as CI


def test_an_unreadable_metadata_json_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(CI, "_TRITON_CACHE_INDEX", {})
    entry, other = tmp_path / "entry", tmp_path / "other"
    entry.mkdir(), other.mkdir()
    (entry / "a_not_json.json").write_text("{not json")
    (entry / "kernel.json").write_text(json.dumps({"hash": "k1", "name": "triton_poi_0", "num_warps": 4, "num_stages": 2}))
    (other / "binary.json").write_bytes(b"\xff\xfe\x00")
    assert CI._triton_cache_files("k1") == {"dir": str(entry), "meta": {"name": "triton_poi_0", "num_warps": 4, "num_stages": 2}}
    assert CI._triton_cache_files("k2") == {}
