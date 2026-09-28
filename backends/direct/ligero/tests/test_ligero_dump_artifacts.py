"""``bench-vu --dump-dir`` inside the run directory lists the dump tree in the result's ``artifacts`` (research/result/v0.1), so
the store's ``run-files/v1`` preserves the proofs, statements, coins and ``system.bin`` beside the result and a verifier who was
not the producer can re-check them (lane a100-dumps).  ``backends.direct.ligero.vu`` imports the prover (torch): skipped without it."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("torch")

from backends.direct.ligero.vu import dump_artifacts  # noqa: E402


def test_dump_inside_the_run_directory_is_listed_relative_to_it(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    assert dump_artifacts(tmp_path / "proofs", str(tmp_path / "result.json")) == ["proofs"]
    assert dump_artifacts(tmp_path / "dumps" / "zk", None) == ["dumps/zk"]


def test_dump_elsewhere_or_absent_is_not_the_runs_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path / "run"))
    assert dump_artifacts(None, str(tmp_path / "run" / "result.json")) == []
    assert dump_artifacts(tmp_path / "elsewhere", str(tmp_path / "run" / "result.json")) == []
    assert dump_artifacts(tmp_path / "run", None) == []                      # the run directory itself is not an artifact


def test_without_a_run_directory_the_out_files_directory_is_the_run_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    assert dump_artifacts(tmp_path / "r" / "proofs", str(tmp_path / "r" / "result.json")) == ["proofs"]
    assert dump_artifacts(tmp_path / "proofs", str(tmp_path / "r" / "result.json")) == []
    assert dump_artifacts(tmp_path / "proofs", None) == []
