"""The pipeline's best-effort reads fall back on exactly the failures they name: an unreadable or malformed file, a git that
cannot run.  Each case pins the value the caller gets."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from verity_vllm.pipeline import coverage_workloads as gw
from verity_vllm.pipeline import experiment as EXP


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


@pytest.mark.parametrize("export", ["[1, 2]", "{broken", '"a string"'])
def test_code_version_falls_back_to_git_past_an_unusable_export_json(tmp_path: Path, monkeypatch, export: str):
    (tmp_path / "EXPORT.json").write_text(export)
    calls = []

    def run(cmd, **kw):
        calls.append(cmd[:2])
        return subprocess.CompletedProcess(cmd, 0, stdout="abc123\n" if cmd[1] == "rev-parse" else " M verity_vllm/x.py\n", stderr="")

    monkeypatch.setattr(EXP.subprocess, "run", run)
    assert EXP.code_version(tmp_path) == {"source": "git", "commit": "abc123", "dirty": True}
    assert calls == [["git", "rev-parse"], ["git", "status"]]


def test_code_version_is_unknown_when_git_cannot_run(tmp_path: Path, monkeypatch):
    assert EXP.code_version(tmp_path / "no-such-tree") == {"source": "unknown"}

    def timeout(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))

    monkeypatch.setattr(EXP.subprocess, "run", timeout)
    assert EXP.code_version(tmp_path) == {"source": "unknown"}


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
    monkeypatch.setattr(gw, "workload_doc", lambda **kw: seen.append(kw["model"]) or {"note": ""})
    ck = _checkpoints(tmp_path, manifest) if manifest is not None else tmp_path / "missing.json"
    a = gw.Options(case=case, model=model, tokenizer="t", profiles="P", ctx="C", out_dir=str(tmp_path / "out"), checkpoints=str(ck))
    assert gw.main(a) == 0 and seen == [served]
