"""`verity_vllm.observe.capture`'s snapshot index: `Capture._write_snapshot_index` writes `snapshots.json` in the snapshot
directory under the index's schema id, with the snapshot steps in order, the two byte limits, the bytes written and every entry
the capture listed."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from verity_vllm.observe.capture import Capture, ObserverConfig


def test_the_snapshot_index_holds_its_schema_id_the_steps_the_byte_limits_and_every_entry(tmp_path):
    capture = Capture.__new__(Capture)
    capture.cfg = ObserverConfig(snapshot_dir=tmp_path, snapshot_steps=frozenset({7, 3}), snapshot_max_bytes=1 << 20)
    capture._snap_bytes = 12
    capture._snap_index = [{"event": 10, "path": "outs", "file": "values/3/00.bin", "step": 3},
                       {"event": 11, "path": "args[0]", "skipped": "over snapshot_max_bytes", "step": 7}]
    capture._write_snapshot_index()
    doc = json.loads((tmp_path / "snapshots.json").read_text())
    assert sorted(doc) == ["entries", "layout", "note", "schema", "snapshot_max_bytes", "snapshot_max_bytes_op", "steps",
                           "total_bytes"]
    assert doc["schema"] == "verity-capture/snapshots/v1" and doc["layout"] == "spanned-storage-bytes"
    assert doc["steps"] == [3, 7] and doc["total_bytes"] == 12
    assert doc["snapshot_max_bytes"] == 1 << 20 and doc["snapshot_max_bytes_op"] is None
    assert doc["entries"] == capture._snap_index
