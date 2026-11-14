"""`verity_vllm.pipeline.tp.serving_rows` where it takes a block or a list off a document it reads: a row's commit block, that
block's ranks, the versions' command line and replay block, the deferred record's bundles (`record`), and a rank's weight
registration (`from_bundle`).  Each such read takes a field that is absent, null and empty alike, and each case here writes the
field those three ways and holds one answer for all three.  CPU-only: the record is written by hand, and `from_bundle`'s readers
are the stand-ins of `test_tp_serving_rows_ranks`."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.pipeline import test_tp_serving_rows_ranks as RK
from verity_vllm.pipeline import cpu_replay as CPU
from verity_vllm.pipeline.tp import c2 as C2
from verity_vllm.pipeline.tp import commit_replay as CR
from verity_vllm.pipeline.tp import run_root as RT
from verity_vllm.pipeline.tp import serving_rows as SRT

FORMS = ["absent", "null", "empty"]
TAG = RT.TP_ROOT_TAG.decode()
ROOTS = {0: bytes([0x0a]) * 64, 1: bytes([0x0b]) * 64}
TP_ROOT = RT.combine(TAG, 2, ROOTS).hex()
COMMIT = {"tp_run_root": TP_ROOT, "world": 2, "per_rank": {str(r): {"run_root": x.hex()} for r, x in ROOTS.items()}}
ROW = {"pair": 0, "arm": "instrumented", "label": "the row", "workload_sha256": "5c" * 32, "commit": COMMIT}
OF_RECORD = {"0": ["d0"], "1": ["d1"]}
VERSIONS = {"root_tag": TAG, "sampled_replay": {"of_record": OF_RECORD}, "argv": ["tp-commit", "--workload", "w.json", "--tp", "2"]}
RECORD = {"world": 2, "tp_run_root": TP_ROOT, "roots": {str(r): x.hex() for r, x in ROOTS.items()}, "root_tag": TAG, "bundles": {},
          "of_record": OF_RECORD, "label": "the row", "workload": "w.json", "workload_sha256": "5c" * 32}


def _with(doc: dict[str, Any], name: str, form: str, empty: Any) -> dict[str, Any]:
    """``doc`` with its field ``name`` absent, null or ``empty``."""
    rest = {k: v for k, v in doc.items() if k != name}
    return rest if form == "absent" else {**rest, name: None if form == "null" else empty}


def _commit(tmp_path: Path, rows: list[dict[str, Any]], versions: dict[str, Any], deferred: dict[str, Any] | None = None) -> Path:
    """A TP Commit's dir as `record` reads it: its rows, its versions and, for a Commit that deferred, its deferred record."""
    d = tmp_path / "commit"
    d.mkdir()
    (d / "runs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    (d / "versions.json").write_text(json.dumps(versions))
    if deferred is not None:
        (d / C2.DEFERRED).write_text(json.dumps(deferred))
    return d


def test_the_record_of_a_commit_that_deferred_nothing(tmp_path):
    assert SRT.record(_commit(tmp_path, [{"pair": 0, "arm": "control", "label": "the row"}, ROW], VERSIONS)) == RECORD


@pytest.mark.parametrize("form", FORMS)
def test_a_later_row_that_holds_no_commit_block_is_passed_over_for_the_last_one_with_a_tp_run_root(tmp_path, form):
    later = _with({**ROW, "label": "a later run"}, "commit", form, {})
    assert SRT.record(_commit(tmp_path, [ROW, later], VERSIONS)) == RECORD


@pytest.mark.parametrize("form", FORMS)
def test_a_record_whose_one_instrumented_row_holds_no_commit_block_has_no_run_to_read(tmp_path, form):
    d = _commit(tmp_path, [_with(ROW, "commit", form, {})], VERSIONS)
    with pytest.raises(LookupError) as refused:
        SRT.record(d)
    assert str(refused.value) == f"{d / 'runs.jsonl'}: no pair-0 instrumented run with a TP run root (the Commit failed before its roots)"


@pytest.mark.parametrize("form", FORMS)
def test_a_commit_block_that_names_no_rank_is_refused_as_not_the_world(tmp_path, form):
    d = _commit(tmp_path, [{**ROW, "commit": _with(COMMIT, "per_rank", form, {})}], VERSIONS)
    with pytest.raises(ValueError) as refused:
        SRT.record(d)
    assert str(refused.value) == "the record's ranks [] are not the world of 2"


@pytest.mark.parametrize("form", FORMS)
def test_versions_that_hold_no_command_line_name_no_workload(tmp_path, form):
    assert SRT.record(_commit(tmp_path, [ROW], _with(VERSIONS, "argv", form, []))) == {**RECORD, "workload": None}


@pytest.mark.parametrize("form", FORMS)
def test_a_deferred_record_that_lists_no_bundle_gives_none(tmp_path, form):
    deferred = _with({"world": 2, "bundles": [{"pair": 0, "rank": 1, "manifest_sha256": "5a" * 32}]}, "bundles", form, [])
    assert SRT.record(_commit(tmp_path, [ROW], VERSIONS, deferred)) == RECORD


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("name, doc", [("sampled_replay", VERSIONS), ("of_record", VERSIONS["sampled_replay"])],
                         ids=["no replay block", "a replay block of no Programs of record"])
def test_versions_that_name_no_programs_of_record_give_none(tmp_path, name, doc, form):
    less = _with(doc, name, form, {})
    versions = less if name == "sampled_replay" else {**VERSIONS, "sampled_replay": less}
    assert SRT.record(_commit(tmp_path, [ROW], versions)) == {**RECORD, "of_record": {}}


@pytest.mark.parametrize("block, root", [
    ({}, None), ({"weight_registration": None}, None), ({"weight_registration": {}}, None), ({"weight_registration": {"root": "0d" * 64}}, "0d" * 64),
], ids=[*FORMS, "a registration"])
def test_a_ranks_bundle_is_held_to_the_weights_root_its_commit_block_registers_and_to_none_where_it_registers_none(tmp_path, monkeypatch,
                                                                                                                   block, root):
    d, calls, made = RK._deferred(tmp_path, monkeypatch)
    held: list[Any] = []

    def of_record(out: Path, dd: dict[str, Any], b: dict[str, Any], pair: int, rank: int, *, awaiting: bool = True) -> tuple:
        return {}, {"run_root": RK.CLAIMED["1"], **block}, made.store, made.run, {"m": 1}

    def on_read(bundle: Path, tree: dict[str, Any], root_of_record: Any, ck: Any, **kw: Any) -> dict[str, Any]:
        held.append(root_of_record)
        return {}

    monkeypatch.setattr(CR, "_of_record", of_record)
    monkeypatch.setattr(CPU, "OnRead", on_read)
    assert SRT.from_bundle(d, 1, "p.json", tmp_path / "w") == RK.WINDOW and held == [root]
    assert callable(calls["window"][1]["weights"])
