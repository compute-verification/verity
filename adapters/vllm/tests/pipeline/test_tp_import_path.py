"""`verity_vllm.pipeline.tp.import_path.write_import_path_runtime`: the import path of record a TP Commit writes from its running
processes is the driver's repo-local files with every rank's, and a rank that answers nothing or answers without its files, or the
ranks' call failing, leaves the record an error: never a path with a rank's files missing."""
from __future__ import annotations

import json
from typing import Any

import pytest

from verity_vllm.pipeline.tp import rank_worker as W
from verity_vllm.pipeline.tp.import_path import write_import_path_runtime

DRIVER: dict[str, Any] = {"files": ["adapters/vllm/verity_vllm/pipeline/tp/commit.py", "verity/circuits/program.py"], "n": 2, "method": "sys.modules of the running process"}
RANK_0: dict[str, Any] = {"files": ["adapters/vllm/verity_vllm/pipeline/tp/rank_worker.py", "verity/circuits/program.py"], "n": 2, "method": DRIVER["method"], "rank": 0}
RANK_1: dict[str, Any] = {"files": ["adapters/vllm/verity_vllm/pipeline/tp/rank_replay.py"], "n": 1, "method": DRIVER["method"], "rank": 1}


class _LLM:
    def __init__(self, replies) -> None:
        self.replies = replies
        self.rpcs: list[tuple] = []

    def collective_rpc(self, method, timeout=None):
        self.rpcs.append((method, timeout))
        if isinstance(self.replies, BaseException):
            raise self.replies
        return self.replies


def _write(tmp_path, monkeypatch, replies) -> dict:
    monkeypatch.setattr(W, "repo_modules_loaded", lambda: dict(DRIVER))
    llm = _LLM(replies)
    write_import_path_runtime(llm, tmp_path, 60.0)
    assert llm.rpcs == [("tp2_repo_modules", 60.0)]
    return json.loads((tmp_path / "import_path_runtime.json").read_text())


def test_the_import_path_of_record_is_the_drivers_files_with_every_ranks(tmp_path, monkeypatch, caplog):
    doc = _write(tmp_path, monkeypatch, [RANK_0, RANK_1])
    union = sorted({*DRIVER["files"], *RANK_0["files"], *RANK_1["files"]})
    assert {k: v for k, v in doc.items() if k != "schema"} == {"method": DRIVER["method"], "driver": DRIVER, "ranks": [RANK_0, RANK_1], "union_n": 4,
                                                                "union": union}
    assert doc["schema"] == "verity/tp2/import-path-runtime/v1"
    assert caplog.messages == ["[tp2commit] import path of record (runtime): driver 2 files, ranks [2, 1], union 4 -> import_path_runtime.json"]


def test_a_rank_that_imported_no_repo_file_adds_none(tmp_path, monkeypatch):
    doc = _write(tmp_path, monkeypatch, [RANK_0, dict(RANK_1, files=[], n=0)])
    assert doc["union"] == sorted({*DRIVER["files"], *RANK_0["files"]}) and doc["union_n"] == 3


@pytest.mark.parametrize("replies", [[RANK_0, None], [RANK_0, {"n": 0, "rank": 1}], RuntimeError("rank 1 did not answer")])
def test_a_rank_that_answers_nothing_or_without_its_files_leaves_the_record_an_error(tmp_path, monkeypatch, caplog, replies):
    doc = _write(tmp_path, monkeypatch, replies)
    assert set(doc) == {"schema", "error"} and doc["error"] and caplog.messages == []
    assert doc["schema"] == "verity/tp2/import-path-runtime/v1"
