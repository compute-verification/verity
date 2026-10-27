"""`SnapshotStore` and the capture's `skipped` index entries.

A tensor over the snapshot cap (D87 `--snapshot-max-bytes`, or the lower dispatcher-call cap `--snapshot-max-bytes-op`)
is listed in `snapshots.json` with its view and a `skipped` reason and NO `file`.  The store must keep such entries out
of `entries` / `by_alloc` (a file-less Snapshot resolved to the capture root; `_raw` then raised IsADirectoryError the
first time a boundary value lived in a skipped alloc — OLMoE B8-static C256, 1300-token step 0) and expose them as
`skipped` for the report.
"""
from __future__ import annotations

import json

import numpy as np

from verity_vllm.check import fold_compare as cmp
from verity_vllm.observe.events import ViewDesc


def _index(root, entries):
    (root / "snapshots.json").write_text(json.dumps({"layout": "spanned-storage-bytes", "snapshot_max_bytes": 32 << 20,
                                                     "snapshot_max_bytes_op": 24 << 20, "entries": entries}))


def test_skipped_entries_are_kept_out_of_the_store(tmp_path):
    (tmp_path / "values" / "0").mkdir(parents=True)
    kept = ViewDesc(alloc=7, offset=0, shape=(4,), strides=(1,), dtype="uint32", device="cuda:0")
    np.arange(4, dtype=np.uint32).tofile(tmp_path / "values" / "0" / "e1.bin")
    dropped = ViewDesc(alloc=9, offset=0, shape=(6,), strides=(1,), dtype="uint32", device="cuda:0")
    _index(tmp_path, [
        {"event": 1, "path": "outs", "view": kept.to_json(), "dtype": "uint32", "shape": [4], "strides": [1], "byte_offset": 0,
         "nbytes": 16, "layout": "spanned-storage-bytes", "file": "values/0/e1.bin", "step": 0},
        {"event": 2, "path": "outs", "view": dropped.to_json(), "dtype": "uint32", "shape": [6],
         "skipped": "spanned bytes 24 > snapshot_max_bytes 16 (snapshot_max_bytes_op: dispatcher-call event; Triton launch events keep 32)"},
    ])
    store = cmp.SnapshotStore(tmp_path)
    assert [s.event for s in store.entries] == [1] and 9 not in store.by_alloc and 2 not in store.by_event
    assert len(store.skipped) == 1 and store.skipped[0]["event"] == 2 and store.skipped[0]["skipped"].startswith("spanned bytes")
    assert store.meta["snapshot_max_bytes_op"] == 24 << 20
    # the kept one still reads; a lookup in the skipped alloc finds nothing rather than a directory
    assert store.words(store.entries[0]).tolist() == [0, 1, 2, 3]
    assert store.by_alloc.get(9, []) == []
