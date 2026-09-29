"""golden.py (protected): the frozen corpus re-folds to its recorded attribution + digest; a changed attribution fails."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity_vllm.properties import golden

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not (REPO / "data" / "logs" / "m1.jsonl.gz").exists(), reason="corpus logs not present")


def test_corpus_check_passes():
    rep = golden.check(only=["smollm2-135m-m1"])
    assert rep["status"] == "pass" and rep["n_pass"] == 1, rep


def test_changed_attribution_fails(tmp_path):
    corpus = golden.load_corpus()
    e = dict(next(x for x in corpus["entries"] if x["name"] == "smollm2-135m-m1"))
    e["attribution_sha"] = "0" * 64
    (tmp_path / "corpus.json").write_text(json.dumps({"schema": corpus["schema"], "entries": [e]}))
    rep = golden.check(tmp_path / "corpus.json")
    assert rep["status"] == "fail" and "attribution changed" in rep["entries"][0]["reason"]


@pytest.mark.slow
def test_attribution_is_deterministic_and_names_patterns():
    m1 = golden.measure(REPO / "data" / "logs" / "m1.jsonl.gz", "vllm_d9105ea80_sm89_eager")
    m2 = golden.measure(REPO / "data" / "logs" / "m1.jsonl.gz", "vllm_d9105ea80_sm89_eager")
    assert m1["attribution_sha"] == m2["attribution_sha"] and m1["digest"] == m2["digest"]
    assert "gemm.matmul_kernel_persistent" in m1["patterns"]
