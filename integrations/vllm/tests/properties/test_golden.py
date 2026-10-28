"""golden.py (protected): the frozen corpus re-folds to its recorded attribution + digest; a changed attribution fails."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from verity_vllm.properties import golden

REPO = Path(__file__).resolve().parents[2]
needs_corpus = pytest.mark.skipif(not (REPO / "data" / "logs" / "m1.jsonl.gz").exists(), reason="corpus logs not present")


@pytest.mark.slow
@needs_corpus
def test_corpus_check_passes():
    rep = golden.check(only=["smollm2-135m-m1"])
    assert rep["status"] == "pass" and rep["n_pass"] == 1, rep


@pytest.mark.slow
@needs_corpus
def test_changed_attribution_fails(tmp_path):
    corpus = golden.load_corpus()
    e = dict(next(x for x in corpus["entries"] if x["name"] == "smollm2-135m-m1"))
    e["attribution_sha"] = "0" * 64
    (tmp_path / "corpus.json").write_text(json.dumps({"schema": corpus["schema"], "entries": [e]}))
    rep = golden.check(tmp_path / "corpus.json")
    assert rep["status"] == "fail" and "attribution changed" in rep["entries"][0]["reason"]


@pytest.mark.slow
@needs_corpus
def test_attribution_is_deterministic_and_names_patterns():
    m1 = golden.measure(REPO / "data" / "logs" / "m1.jsonl.gz", "vllm_d9105ea80_sm89_eager")
    m2 = golden.measure(REPO / "data" / "logs" / "m1.jsonl.gz", "vllm_d9105ea80_sm89_eager")
    assert m1["attribution_sha"] == m2["attribution_sha"] and m1["digest"] == m2["digest"]
    assert "gemm.matmul_kernel_persistent" in m1["patterns"]


def test_an_absent_log_is_fetched_and_one_the_fetch_cannot_give_is_skipped_with_why(tmp_path, monkeypatch):
    entry = {"name": "x", "log": "data/logs/no-such.jsonl.gz", "profile": "p", "digest": "d", "attribution_sha": "a"}
    (tmp_path / "corpus.json").write_text(json.dumps({"schema": "verity-gen/golden/v1", "entries": [entry]}))
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 1, "", "MISSING  data/logs/no-such.jsonl.gz")
    monkeypatch.setattr(golden.subprocess, "run", run)
    rep = golden.check(tmp_path / "corpus.json")
    assert calls[0][1:5] == ["-m", "research", "data", "fetch-fixtures"] and calls[0][-1] == str(REPO / entry["log"])
    assert rep["status"] == "skipped" and "fetch-fixtures rc 1: MISSING" in rep["entries"][0]["reason"]
