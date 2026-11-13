"""`verity_vllm.observe.capture_log`: a capture log's RunHeader and StepStart rows, and the request each prompt parameter is, on
logs and headers that leave an optional field out or hold an empty value there."""
from __future__ import annotations

import json
from typing import Any

import pytest

from verity_vllm.observe import capture_log as CL


def _log(tmp_path, *steps: dict[str, Any]) -> str:
    """A log's lines as the writer orders them: `$e` first, the RunHeader, then each StepStart."""
    lines = [{"$e": "RunHeader", "workload": {"requests": [{"request_id": "a"}]}}, *({"$e": "StepStart", **s} for s in steps)]
    path = tmp_path / "capture.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return str(path)


def test_a_step_with_no_logits_indices_or_an_empty_value_there_reads_as_none(tmp_path):
    rows = {"rows": [["a", 0], ["a", 1]], "logits_rows": [["a", 1]]}
    steps: list[dict[str, Any]] = [{"step": 0, **rows, "logits_indices": [1, 3]}, {"step": 1, **rows}, {"step": 2, **rows, "logits_indices": None},
                                   {"step": 3, **rows, "logits_indices": 0}, {"step": 4, **rows, "logits_indices": ""}]
    header, got = CL.read_steps(_log(tmp_path, *steps))
    assert header["workload"] == {"requests": [{"request_id": "a"}]} and sorted(got) == [0, 1, 2, 3, 4]
    assert got[0] == {"rows": [("a", 0), ("a", 1)], "logits_rows": [("a", 1)], "logits_indices": [1, 3]}
    assert [got[s]["logits_indices"] for s in (1, 2, 3, 4)] == [[], [], [], []]
    with pytest.raises(TypeError, match="not iterable"):       # a value that is not empty is read as the list it has to be
        CL.read_steps(_log(tmp_path, {"step": 0, **rows, "logits_indices": 7}))


def test_a_headers_requests_are_its_prompt_parameters_and_a_child_reads_its_parents():
    reqs = [{"request_id": "a"}, {"request_id": "b"}]
    header = {"workload": {"requests": reqs, "children": [{"request_id": "c", "parent": "b"}]}}
    assert CL.request_params(header) == (reqs, {"a": 0, "b": 1, "c": 1}, {"a": "a", "b": "b", "c": "b"})
    childless: tuple[dict[str, Any], ...] = ({}, {"children": None}, {"children": []}, {"children": 0}, {"children": ""})
    for children in childless:
        assert CL.request_params({"workload": {"requests": reqs, **children}}) == (reqs, {"a": 0, "b": 1}, {"a": "a", "b": "b"}), children
    headers: tuple[dict[str, Any], ...] = ({}, {"workload": None}, {"workload": 0}, {"workload": {}}, {"workload": {"requests": None}},
                                           {"workload": {"requests": 0}}, {"workload": {"requests": []}})
    for none in headers:
        with pytest.raises(ValueError, match="RunHeader.workload has no requests"):
            CL.request_params(none)
    with pytest.raises(ValueError, match="child 'c' names unknown parent 'z'"):
        CL.request_params({"workload": {"requests": reqs, "children": [{"request_id": "c", "parent": "z"}]}})
