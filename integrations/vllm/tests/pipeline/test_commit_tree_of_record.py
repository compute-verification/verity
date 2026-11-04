"""The TP Commit's tree of record is the one code identity (`source_identity.code_identity`): the import closure of the repository it
runs from -- the integration plus Verity core -- by content."""
from __future__ import annotations

from pathlib import Path

from verity_vllm.pipeline import source_identity
from verity_vllm.pipeline.tp import commit as tp_commit

CORE = Path("verity")


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / CORE / "ir").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname = 'verity'\n")
    (repo / CORE / "__init__.py").write_text("")
    (repo / CORE / "ir" / "codec.py").write_text("def _spec_id(fn): return fn\n")
    for d in source_identity.core_dirs(source_identity.TREE):
        (repo / d).mkdir(parents=True, exist_ok=True)
        (repo / d / "__init__.py").touch()
    (repo / "integrations" / "vllm" / "verity_vllm" / "pipeline" / "tp").mkdir(parents=True)
    (repo / "integrations" / "vllm" / "verity_vllm" / "__init__.py").write_text("")
    (repo / "integrations" / "vllm" / "verity_vllm" / "pipeline" / "tp" / "commit.py").write_text("TAG = 1\n")
    return repo


def test_tree_of_record_is_the_code_identity_and_follows_core_edits(tmp_path):
    repo = _repo(tmp_path)
    a = tp_commit.tree_of_record(repo)
    assert a["sha256"] == source_identity.code_identity(repo)["sha256"] and a["closure"] == source_identity.closure()
    (repo / CORE / "ir" / "codec.py").write_text("def _spec_id(fn): return (fn,)\n")
    assert tp_commit.tree_of_record(repo)["sha256"] != a["sha256"], "an edited core file is another tree of record"


def test_tree_of_record_defaults_to_the_repository_it_runs_from():
    rec = tp_commit.tree_of_record()
    assert rec["root"] == str(source_identity.TREE) and rec["sha256"] == source_identity.code_identity(source_identity.TREE)["sha256"]
