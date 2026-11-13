"""`verity_vllm.acquire.committer_api.declaration`: the coverage declaration a committer emits, with the schema id its records
on disk carry."""
from __future__ import annotations

from verity_vllm.acquire import committer_api as API


def test_a_declaration_carries_its_schema_id_and_what_it_was_given():
    classes = [{"class": "weights", "how": "registered-input", "rule": "registration", "count": 1, "bytes": 8, "evidence": ""}]
    assert API.declaration("native_host", "eager", classes) == {
        "schema": "verity/commit-integ/coverage-declaration/v1", "committer": "native_host", "path": "eager",
        "classes": classes, "notes": []}
    assert API.declaration("native_host", "eager", classes, ["a note"])["notes"] == ["a note"]
