"""The lease record's id (`verity_vllm.pipeline.gpu_lease.run`): what `run` returns and the file it writes both carry the id
on disk, and it is the constant `records.schema_ids` holds."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import verity_vllm.pipeline.gpu_lease as GL
from verity_vllm.records import schema_ids


def never_leased(row_dir: Path) -> tuple[int, dict[str, Any]]:
    """A Commit whose lease tool ends before a GPU is held: `run` writes the record all the same."""
    return GL.run(str(row_dir), str(row_dir / "commit.log"), ["true"], 60, {"GPU_LEASE": "/l/gpu-lease"},
                  lambda _timeout, _argv: 75, clock=lambda: 0.0)


def test_the_lease_record_carries_the_id_on_disk_in_what_run_returns_and_in_its_file(tmp_path: Path) -> None:
    rc, rec = never_leased(tmp_path)
    assert rc == 75
    assert rec["schema"] == "verity-vllm/gpu-lease/v1"
    assert json.loads((tmp_path / GL.RECORD).read_text())["schema"] == "verity-vllm/gpu-lease/v1"


def test_the_lease_record_carries_the_id_the_records_package_holds(tmp_path: Path) -> None:
    """The record's id is written once, in `records.schema_ids`: its writer here and a reader elsewhere cannot hold two."""
    assert GL.SCHEMA is schema_ids.GPU_LEASE_SCHEMA
    assert never_leased(tmp_path)[1]["schema"] is schema_ids.GPU_LEASE_SCHEMA
