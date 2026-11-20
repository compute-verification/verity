"""The pipeline's best-effort reads fall back on exactly the failures they name: an unreadable or malformed file.  Each case
pins the value the caller gets; `experiment.code_version`'s fallbacks to git are test_experiment_index's.  A file nested past
what Python's parser reads is malformed too: `json` raises `RecursionError` for it, which is no `ValueError`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity_vllm.pipeline import build as B
from verity_vllm.pipeline import experiment as EXP
from verity_vllm.pipeline import global_program as GP
from verity_vllm.pipeline import match as RC

NESTED = "[" * 100_000 + "]" * 100_000
UNPARSED = pytest.mark.parametrize("text", ["{not JSON", NESTED], ids=["not JSON", "nested past the parser"])


@UNPARSED
def test_a_component_whose_result_cannot_be_parsed_carries_no_target_and_no_histogram(tmp_path: Path, text: str):
    (tmp_path / "result.json").write_text(text)
    assert GP.component_declared_target(str(tmp_path)) is None and GP.component_spec_histogram(str(tmp_path)) is None




@UNPARSED
def test_a_workload_that_cannot_be_parsed_is_no_workload(tmp_path: Path, text: str):
    (tmp_path / "wl.json").write_text(text)
    assert RC._workload_doc(str(tmp_path / "wl.json")) is None


@UNPARSED
def test_a_build_s_manifest_takes_an_inputs_trace_it_cannot_parse_as_none_written(tmp_path: Path, text: str):
    res = {"utc": "t", "wrapper": "step", "LP": 1, "T": 1, "ok": True, "program": {"digest": "dg"}}
    B._write_manifest(str(tmp_path), dict(res))
    none_written = (tmp_path / "construction_manifest.json").read_text()
    (tmp_path / "inputs.json").write_text(text)
    B._write_manifest(str(tmp_path), dict(res))
    assert (tmp_path / "construction_manifest.json").read_text() == none_written
    (tmp_path / "inputs.json").write_text(json.dumps({"n_open_events": 3}))
    B._write_manifest(str(tmp_path), dict(res))
    assert json.loads((tmp_path / "construction_manifest.json").read_text())["inputs_trace"]["n_open_events"] == 3


@pytest.mark.parametrize("content", [b"{not json", b"\xff\xfe\x00", None])
def test_load_reports_an_unreadable_report_as_absent(tmp_path: Path, content: bytes | None):
    if content is None:
        (tmp_path / "r.json").mkdir()
    else:
        (tmp_path / "r.json").write_bytes(content)
    assert EXP._load(tmp_path, "r.json") is None
    assert EXP._load(tmp_path, "missing.json") is None


def test_load_reads_a_report(tmp_path: Path):
    (tmp_path / "r.json").write_text('{"a": [1, 2]}')
    assert EXP._load(tmp_path, "r.json") == {"a": [1, 2]}


def test_code_version_reads_export_json(tmp_path: Path):
    (tmp_path / "EXPORT.json").write_text(json.dumps({"commit": "c0ffee", "patch_sha256": "ab", "untracked": 2, "fallback": False}))
    assert EXP.code_version(tmp_path) == {"source": "EXPORT.json", "commit": "c0ffee", "dirty_patch_sha256": "ab", "untracked": 2,
                                          "fallback": False}


def test_write_spec_leaves_an_unreadable_spec_alone(tmp_path: Path):
    (tmp_path / "spec.json").write_text("{broken")
    assert EXP.write_spec(tmp_path, {"digest": "d1"}) == {"written": False, "existing_digest": None, "would_be_digest": "d1",
                                                          "same": False}
    assert (tmp_path / "spec.json").read_text() == "{broken"


