"""The Commit binds the manifest's partition of record (`binding.partition_of_record`): under Q_word the manifest names each Program's
`verity/partition/v1` object and digest, and the Commit refuses, before any engine, a partition that is malformed, carries another
digest, or names a Program that is not of record, and a manifest that records no passing strict word check for every part."""
from __future__ import annotations

import json
import logging
import re
import subprocess
import sys

import pytest

from verity_vllm.commit.partition import (
    partition_entry,
    partition_of_record,
    program_digests,
    require_partition,
    word_check_refusal,
)
from verity_vllm.pipeline import research_tools
from verity_vllm.query.compose import _with_partitions
from verity_vllm.query.required import (
    Q_MODULE_BODY_ID,
    Q_WORD_ID,
    QUERY_OF_RECORD,
    partition_of,
)

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


def _checked(query=Q_WORD_ID, **kw) -> dict:
    return {"query": query, "strict": True, "ok": True, "violations": 0, "calls": 9, "units": 40, "gates": 900, **kw}


def test_the_commit_takes_a_q_word_manifest_that_records_a_passing_word_check_for_every_part():
    one = _man(partition_of(SHA_A))
    one["query"]["word_check"] = _checked()
    assert require_partition(one, None, None) is one and word_check_refusal(one["query"]) is None
    q = _with_partitions({"query_id": Q_WORD_ID}, "by_component", {"r0": {"partition": partition_of(SHA_A), "word_check": _checked()},
                                                                  "r1": {"partition": partition_of(SHA_B), "word_check": _checked()}})
    assert q["word_check"] == {"by_component": {"r0": _checked(), "r1": _checked()}} and word_check_refusal(q) is None
    assert word_check_refusal(_with_partitions(dict(q), "by_rank", {"0": q, "1": q})) is None
    body = _man(query=Q_MODULE_BODY_ID)                                  # A/B only: no partition, and no word check is required
    assert require_partition(body, None, None) is body


def test_the_commit_logs_the_partition_it_checked_and_prints_nothing(caplog, capsys):
    one = _man(partition_of(SHA_A))
    one["query"]["word_check"] = _checked()
    caplog.set_level(logging.INFO, logger="verity_vllm")
    require_partition(one, None, None)
    require_partition(_man(query=Q_MODULE_BODY_ID), None, None)
    said = [r.getMessage() for r in caplog.records if r.name == "verity_vllm.commit.partition"]
    assert len(said) == 2 and all(m.startswith("[delta] partition of record: {") for m in said)
    assert said[0].endswith("} (word check recorded and passed)") and said[1].endswith("}")
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("check, why", [
    (None, "no word check recorded"),
    (_checked(ok=False, violations=1), "ok=False violations=1"),
    (_checked(strict=False), "strict=False"),
    (_checked(query="Q_word_v1{X=8,W=32,R=no-recompute}"), "X=8"),
    ({"by_component": {"r0": _checked()}}, "by_component/r1: no word check"),
    ({"by_component": {"r0": {"nope": 1}}}, "neither a word check nor a composition"),
])
def test_the_commit_refuses_a_q_word_manifest_without_a_passing_word_check_by_name(check, why):
    parts = {"r0": {"partition": partition_of(SHA_A)}, "r1": {"partition": partition_of(SHA_B)}}
    if isinstance(check, dict) and "by_component" in check:
        q = dict(_with_partitions({"query_id": Q_WORD_ID}, "by_component", parts), word_check=check)
    else:
        q = dict(_man(partition_of(SHA_A))["query"], **({"word_check": check} if check else {}))
    assert why in word_check_refusal(q)
    with pytest.raises(SystemExit, match=re.escape(why)):
        require_partition({"query": q}, None, None)


def test_the_commit_reads_the_programs_the_builds_recorded(tmp_path):
    for name, sha in (("build_request_LP8_T2", SHA_A), ("build/rank1/build_request_LP8_T2", SHA_B), ("build_step", None)):
        d = tmp_path / name
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({"program": {"digest": sha} if sha else {}}))
    assert program_digests(str(tmp_path), None) == {SHA_A, SHA_B}


def test_a_result_without_a_program_block_names_no_digest(tmp_path):
    """`verity_vllm.commit.partition.program_digests`: a Build's `result.json` that leaves the `program` block out, holds it empty or
    null, holds something that is no object, or is not JSON, names no Program; the others' digests are read all the same."""
    docs = {"none": "{}", "null": '{"program": null}', "empty": '{"program": {}}', "text": '{"program": "x"}', "torn": "{",
            "no_digest": '{"program": {"sha": "x"}}', "named": json.dumps({"program": {"digest": SHA_A}})}
    for name, text in docs.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "result.json").write_text(text)
    assert program_digests(str(tmp_path)) == {SHA_A}
    assert program_digests(None, "") == set()


@pytest.mark.parametrize("man", [{}, {"query": None}, {"query": {}}, {"query": {"partition": partition_of(SHA_A)}}])
def test_a_manifest_without_a_query_header_is_not_of_the_query_of_record(man):
    """`verity_vllm.commit.partition`: a manifest with no `query` header, or one that names no query, binds no partition and needs no
    word check; `partition_of_record` says so and `require_partition` hands the manifest back."""
    assert partition_of_record(man, {SHA_A}) == {"query": None, "checked": False, "why": "not the query of record: no partition to bind"}
    assert require_partition(man, None, None) is man


def test_the_binding_maps_partition_entry_is_the_manifests_partition_or_nothing():
    """`verity_vllm.commit.partition.partition_entry`: no manifest, a manifest without a `query` header and a header without a
    partition give no entry; a header that names one gives it, whatever its query."""
    part = partition_of(SHA_A)
    blank: list[dict | None] = [None, {}, {"query": None}, {"query": {}}, _man(), _man(query=Q_MODULE_BODY_ID), {"query": {"partition": None}}]
    for man in blank:
        assert partition_entry(man) == {}
    assert partition_entry(_man(part)) == {"partition": part} == partition_entry({"query": {"partition": part}})


@pytest.mark.parametrize("other", [{}, {"by_rank": None}, {"by_rank": {}}])
def test_a_composition_names_its_parts_by_component_by_rank_or_both(other):
    """`verity_vllm.commit.partition`: a composition that holds `by_component` alone, or with a `by_rank` that is null or empty, has
    the parts of `by_component`; one that holds both has the parts of both."""
    a, b = partition_of(SHA_A), partition_of(SHA_B)
    rec = partition_of_record(_man({"by_component": {"r0": a, "r1": b}, **other}), {SHA_A, SHA_B})
    assert rec["partitions"] == 2 and rec["digests"] == sorted([a["digest"], b["digest"]])
    both = partition_of_record(_man({"by_component": {"r0": a}, "by_rank": {"1": b}}), {SHA_A, SHA_B})
    assert both["partitions"] == 2
    assert partition_of_record(_man({"by_rank": {"0": a}, "by_component": None}), {SHA_A})["partitions"] == 1
    with pytest.raises(SystemExit, match="by_rank/1: names Program"):
        partition_of_record(_man({"by_component": {"r0": a}, "by_rank": {"1": b}}), {SHA_A})


def test_every_entry_point_records_the_query_of_record():
    import dataclasses

    from tests.regression.checks import attempt_provenance
    from verity_vllm.pipeline import row_records
    from verity_vllm.pipeline.row import _Row
    row_default = next(f.default for f in dataclasses.fields(_Row) if f.name == "query")
    assert QUERY_OF_RECORD == Q_WORD_ID == research_tools.DEFAULT_QUERY == attempt_provenance.DEFAULT_QUERY == row_default == row_records.QUERY_OF_RECORD


def test_importing_the_module_loads_neither_numpy_nor_the_core():
    """`verity_vllm.commit.partition`: `pipeline.row_records` imports `word_check_refusal` from it, so what this module loads at import
    every module of the row driver loads.  Only `partition_of_record`, when called, reads the partition object's module."""
    code = ("import sys; before = set(sys.modules); import verity_vllm.commit.partition; "
            "print(sorted({m.split('.')[0] for m in set(sys.modules) - before} & {'numpy', 'torch', 'verity'}))")
    child = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "[]"
