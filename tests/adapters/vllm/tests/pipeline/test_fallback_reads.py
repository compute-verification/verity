"""The pipeline's best-effort reads fall back on exactly the failures they name: an unreadable or malformed file.  Each case
pins the value the caller gets; `experiment.code_version`'s fallbacks to git are test_experiment_index's.  A file nested past
what Python's parser reads is malformed too: `json` raises `RecursionError` for it, which is no `ValueError`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity_vllm.pipeline import build as B
from verity_vllm.pipeline import coverage_workloads as gw
from verity_vllm.pipeline import experiment as EXP
from verity_vllm.pipeline import global_program as GP
from verity_vllm.pipeline import manifest as MAN
from verity_vllm.pipeline import match as RC

NESTED = "[" * 100_000 + "]" * 100_000
UNPARSED = pytest.mark.parametrize("text", ["{not JSON", NESTED], ids=["not JSON", "nested past the parser"])


@UNPARSED
def test_a_component_whose_result_cannot_be_parsed_carries_no_target_and_no_histogram(tmp_path: Path, text: str):
    (tmp_path / "result.json").write_text(text)
    assert GP.component_declared_target(str(tmp_path)) is None and GP.component_spec_histogram(str(tmp_path)) is None


@UNPARSED
def test_a_build_directory_whose_result_cannot_be_parsed_is_no_request_program_s(tmp_path: Path, text: str):
    for name, result in (("build_request_a", text), ("build_request_b", json.dumps({"ok": True, "LP": 5, "T": 2, "C": 0}))):
        (tmp_path / name).mkdir()
        (tmp_path / name / "result.json").write_text(result)
    fixture = {"requests": [{"request_id": "r0", "prompt_len": 5, "max_tokens": 3}]}
    assert MAN._program_dirs_for(fixture, str(tmp_path)) == {"r0": str(tmp_path / "build_request_b")}


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


def _checkpoints(tmp_path: Path, doc) -> Path:
    p = tmp_path / "checkpoints.json"
    p.write_text(doc if isinstance(doc, str) else json.dumps(doc))
    return p


def test_checkpoint_repo_reads_the_first_entry_of_the_role(tmp_path: Path):
    p = _checkpoints(tmp_path, {"checkpoints": [{"role": "B1", "repo": "org/b1"}, {"role": "B0", "repo": "org/b0"},
                                                {"role": "B0", "repo": "org/later"}]})
    assert gw.checkpoint_repo(p, "B0") == "org/b0"
    assert gw.checkpoint_repo(str(p), "B1") == "org/b1"
    assert gw.checkpoint_repo(p, "LLAMA") is None


@pytest.mark.parametrize("doc", ["{broken", [1], {"other": []}, {"checkpoints": None}, {"checkpoints": 5}, {"checkpoints": "B0"},
                                 {"checkpoints": {"B0": {"repo": "org/b0"}}}, {"checkpoints": ["B0", {"role": "B0", "repo": "org/b0"}]},
                                 {"checkpoints": [{"role": "B0"}, {"role": "B0", "repo": "org/b0"}]}])
def test_checkpoint_repo_is_none_for_a_file_not_of_the_form(tmp_path: Path, doc):
    assert gw.checkpoint_repo(_checkpoints(tmp_path, doc), "B0") is None


def test_checkpoint_repo_is_none_without_the_file(tmp_path: Path):
    assert gw.checkpoint_repo(tmp_path / "missing.json", "B0") is None
    assert gw.checkpoint_repo(tmp_path, "B0") is None


@pytest.mark.parametrize(("case", "manifest", "model", "served"), [
    ("B0", {"checkpoints": [{"role": "B0", "repo": "org/b0"}]}, None, "org/b0"),
    ("B1", {"checkpoints": [{"role": "B1-B4,R*", "repo": "org/b1"}]}, None, "Qwen/Qwen2.5-1.5B"),
    ("B0", None, None, "HuggingFaceTB/SmolLM2-135M"),
    ("X9", None, None, "unknown"),
    ("B1", None, "org/given", "org/given")])
def test_main_serves_the_manifest_s_model_else_the_role_s(tmp_path: Path, monkeypatch, case, manifest, model, served):
    seen = []
    monkeypatch.setattr(gw, "Tok", lambda path: None)

    def workload_doc(**kw):
        seen.append(kw["model"])
        return {"note": ""}

    monkeypatch.setattr(gw, "workload_doc", workload_doc)
    ck = _checkpoints(tmp_path, manifest) if manifest is not None else tmp_path / "missing.json"
    a = gw.Options(case=case, model=model, tokenizer="t", profiles="P", ctx="C", out_dir=str(tmp_path / "out"), checkpoints=str(ck))
    assert gw.main(a) == 0 and seen == [served]
