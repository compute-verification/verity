"""golden.py (protected): a corpus log absent from the checkout is fetched, and one the fetch can't give is skipped with why. The
frozen corpus itself, whose logs are over a tracked file's limit, is `test_golden_of_record`."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


from verity_vllm.properties import golden

REPO = Path(__file__).resolve().parents[2]


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
