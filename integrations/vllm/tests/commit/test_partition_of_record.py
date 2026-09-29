"""The Commit binds the manifest's partition of record (`binding.partition_of_record`): under Q_word the manifest names each Program's
`verity/partition/v1` object and digest, and the Commit refuses, before any engine, a partition that is malformed, carries another
digest, or names a Program that is not of record."""
from __future__ import annotations

import json

import pytest

from verity_vllm.commit.partition import partition_of_record, program_sha512s
from verity_vllm.pipeline import research_tools
from verity_vllm.query.compose import _with_partitions
from verity_vllm.query.required import Q_MODULE_BODY_ID, Q_WORD_ID, QUERY_OF_RECORD, partition_of

SHA_A, SHA_B = "a" * 128, "b" * 128


def _man(partition=None, query=Q_WORD_ID) -> dict:
    q = {"query_id": query}
    if partition is not None:
        q["partition"] = partition
    return {"query": q}


def test_a_partition_of_record_is_checked_and_recorded():
    rec = partition_of_record(_man(partition_of(SHA_A)), {SHA_A})
    assert rec["checked"] and rec["partitions"] == 1 and rec["programs_checked"] and rec["digests"] == [partition_of(SHA_A)["digest"]]
    assert partition_of_record(_man(partition_of(SHA_A)))["programs_checked"] is False, "no Build records: the object and digest only"


def test_the_module_body_query_binds_no_partition():
    assert partition_of_record(_man(query=Q_MODULE_BODY_ID)) == {"query": Q_MODULE_BODY_ID, "checked": False,
                                                                  "why": "not the query of record: no partition to bind"}


@pytest.mark.parametrize("bad, why", [
    (None, "names no partition"),
    ({"object": partition_of(SHA_A)["object"], "digest": partition_of(SHA_B)["digest"]}, "digest"),
    ({"object": dict(partition_of(SHA_A)["object"], query={"name": "Q_word", "version": 1, "params": {"X": 8, "W": 32}}),
      "digest": partition_of(SHA_A)["digest"]}, "not a verity/partition/v1 object of Q_word v1"),
    (partition_of(SHA_B), "not a Program of record"),
    ({"by_component": {"r0": {"nope": 1}}}, "neither a partition nor a composition"),
])
def test_a_wrong_partition_is_refused_by_name(bad, why):
    with pytest.raises(SystemExit, match=why):
        partition_of_record(_man(bad), {SHA_A})


def test_composed_manifests_name_every_parts_partition():
    q = _with_partitions({"query_id": Q_WORD_ID, "partition": partition_of(SHA_A)}, "by_component",
                         {"r0": {"partition": partition_of(SHA_A)}, "r1": {"partition": partition_of(SHA_B)}})
    assert q["partition"] == {"by_component": {"r0": partition_of(SHA_A), "r1": partition_of(SHA_B)}}
    merged = _with_partitions(dict(q), "by_rank", {"0": q, "1": {"partition": partition_of(SHA_B)}})
    rec = partition_of_record({"query": merged}, {SHA_A, SHA_B})
    assert rec["partitions"] == 3
    with pytest.raises(SystemExit, match="by_rank/0/by_component/r1: names Program"):
        partition_of_record({"query": merged}, {SHA_B} - {SHA_B} | {SHA_A})


def test_the_commit_reads_the_programs_the_builds_recorded(tmp_path):
    for name, sha in (("build_request_LP8_T2", SHA_A), ("build/rank1/build_request_LP8_T2", SHA_B), ("build_step", None)):
        d = tmp_path / name
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({"program": {"digest": "0" * 64, **({"sha512": sha} if sha else {})}}))
    assert program_sha512s(str(tmp_path), None) == {SHA_A, SHA_B}


def test_every_entry_point_records_the_query_of_record():
    from tests.regression.checks import attempt_provenance
    from verity_vllm.pipeline.row import _Row
    import dataclasses
    row_default = next(f.default for f in dataclasses.fields(_Row) if f.name == "query")
    assert QUERY_OF_RECORD == Q_WORD_ID == research_tools.DEFAULT_QUERY == attempt_provenance.DEFAULT_QUERY == row_default
