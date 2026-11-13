"""What `pipeline.commit_plan` reads of a manifest, a loaded Program and a plan that leave a block out: the Program digests of
record and the facts derived under them (`oracle_producers_of_record`), a plan's key (`keyed`) and its producer facts (`use`).
CPU only, over stand-ins for the Program loaders and the oracle's reading; the plan a Commit takes whole is
`test_commit_plan.py`."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import verity_vllm.pipeline.commit_plan as CP
from verity_vllm.query.required import QUERIES

Record = dict[str, Any]
ABSENT = object()
#: what a document may hold where a block is left out: no field, a null, or an empty block
NO_BLOCK: tuple[object, ...] = (ABSENT, None, {})
QUERY = QUERIES[0]
D, E = "d" * 64, "e" * 64
REQS: list[Record] = [{"request_id": "a", "prompt_token_ids": [1, 2, 3], "max_tokens": 4, "cached": 2},
                      {"request_id": "b", "index": 5, "prompt_token_ids": [1]}]
FACTS: dict[tuple[str, str], Record] = {("op", "m0"): {"operand": "x", "consumers": ["c"]}, ("op", "m1"): {"conflict": True}}
DERIVED: Record = {"used": True, "members": 2, "with_operand": 1, "with_consumers": 1, "conflicts": 1}
NOT_THE_QUERY_PATH_S: Record = {"used": False, "why": "manifest is not the query path's: its rows carry the producer annotations"}


def with_(doc: Record, field: str, value: object) -> Record:
    """`doc` with `field` left out (`ABSENT`) or holding `value`."""
    rest = {k: v for k, v in doc.items() if k != field}
    return rest if value is ABSENT else {**rest, field: value}


def manifest(**more: Any) -> Record:
    return {"query": {"query_id": QUERY}} | more


def options(**more: Any) -> SimpleNamespace:
    return SimpleNamespace(**{"program_dir": "/program", "programs_root": None, "program_digest": D} | more)


class Programs:
    """Stand-ins for the manifest format's two Program loaders and the oracle's reading of the loaded Programs, each call kept."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, program: Record) -> None:
        self.program = program
        self.calls: list[tuple[Any, ...]] = []
        monkeypatch.setattr(CP.MF, "load_program", self.load_program)
        monkeypatch.setattr(CP.MF, "programs_for", self.programs_for)
        monkeypatch.setattr(CP.OC, "producers_of_programs", self.producers_of_programs)

    def load_program(self, path: str) -> Record:
        self.calls.append(("load", path))
        return self.program

    def programs_for(self, workload: Record, root: str) -> dict[str, Record]:
        self.calls.append(("for", workload, root))
        return {r["request_id"]: self.program for r in workload["requests"]}

    def producers_of_programs(self, loaded: dict[str, Record], *, aliases: Any, query: str) -> dict[tuple[str, str], Record]:
        self.calls.append(("producers", dict(loaded), aliases, query))
        return dict(FACTS)


def derived(args: SimpleNamespace, required: Record | None, sampling: Record | None = None) -> tuple[Any, Record]:
    """`oracle_producers_of_record` for the two requests, its record without the time it took and the method's text."""
    facts, record = CP.oracle_producers_of_record(args, required, REQS, sampling or {})
    return facts, {k: v for k, v in record.items() if k not in ("wall_s", "method")}


#: which of the four the module's import adds to what a bare interpreter of this environment already holds
LOADED = ("import json, sys\n"
          "held = set(sys.modules)\n"
          "import verity_vllm.pipeline.commit_plan\n"
          "print(json.dumps([p for p in ('numpy', 'torch', 'vllm', 'verity') if p in sys.modules and p not in held]))\n")


def test_the_plan_s_module_loads_neither_numpy_torch_vllm_nor_the_core_at_import() -> None:
    """A process that reads a plan's key or a plan's record (the row's stages, the Commit's parser) loads the module and none of
    the four with it: what derives the facts (`query.required`, the workload's reader) is imported where it is called."""
    out = subprocess.run([sys.executable, "-c", LOADED], capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == []


def test_the_facts_are_read_from_the_one_program_dir_for_every_request(monkeypatch: pytest.MonkeyPatch) -> None:
    program = {"result": {"program_digest": D}}
    programs = Programs(monkeypatch, program)
    assert derived(options(), manifest(op_path_aliases={"a": "b"})) == (FACTS, DERIVED)
    assert programs.calls == [("load", "/program"), ("producers", {"r0": program, "r5": program}, {"a": "b"}, QUERY)]


def test_the_facts_are_read_from_the_programs_root_by_each_request_s_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    program = {"result": {"program_digest": D}}
    programs = Programs(monkeypatch, program)
    assert derived(options(programs_root="/programs"), manifest(), {"max_tokens": 7}) == (FACTS, DERIVED)
    shapes = {"requests": [{"request_id": "r0", "prompt_len": 3, "max_tokens": 4, "cached": 2},
                           {"request_id": "r5", "prompt_len": 1, "max_tokens": 7, "cached": 0}]}
    assert programs.calls == [("for", shapes, "/programs"), ("producers", {"r0": program, "r5": program}, None, QUERY)]


@pytest.mark.parametrize("required", [None, {}, {"query": None}, {"query": {}}, {"query": {"query_id": "another"}}])
def test_a_manifest_the_query_path_did_not_build_has_no_facts_to_derive(monkeypatch: pytest.MonkeyPatch, required: Record | None) -> None:
    programs = Programs(monkeypatch, {"result": {"program_digest": D}})
    assert derived(options(), required) == (None, NOT_THE_QUERY_PATH_S) and programs.calls == []


def test_with_no_program_dir_and_no_programs_root_no_fact_is_derivable(monkeypatch: pytest.MonkeyPatch) -> None:
    programs = Programs(monkeypatch, {"result": {"program_digest": D}})
    assert derived(options(program_dir=None), manifest()) == (
        None, {"used": False, "why": "no --programs-root / --program-dir: producer facts not derivable"})
    assert programs.calls == []


@pytest.mark.parametrize("program", [{"result": {"program_digest": D}, "instances": {"digest": E}}, {"instances": {"digest": D}},
                                     {"result": None, "instances": {"digest": D}}, {"result": {}, "instances": {"digest": D}},
                                     {"result": {"program_digest": None}, "instances": {"digest": D}}])
def test_a_program_s_digest_is_its_build_result_s_and_else_its_instances(monkeypatch: pytest.MonkeyPatch, program: Record) -> None:
    Programs(monkeypatch, program)
    assert derived(options(), manifest()) == (FACTS, DERIVED)


@pytest.mark.parametrize("result", NO_BLOCK)
@pytest.mark.parametrize("instances", NO_BLOCK)
def test_a_program_that_records_no_digest_is_not_of_record(monkeypatch: pytest.MonkeyPatch, result: object, instances: object) -> None:
    programs = Programs(monkeypatch, with_(with_({}, "result", result), "instances", instances))
    assert derived(options(), manifest()) == (
        None, {"used": False, "why": "Program(s) ['r0', 'r5'] not of record (of-record ['dddddddddddddddd'])"})
    assert programs.calls == [("load", "/program")], "the oracle reads no Program that is not of record"


@pytest.mark.parametrize("more", [{"component_digests": [E, D]}, {"components": {"c0": {"program_digest": D}}},
                                  {"component_digests": None, "components": {"c0": "no block", "c1": {}, "c2": {"program_digest": D}}}])
def test_the_manifest_s_component_digests_are_of_record_beside_the_commit_s_own(monkeypatch: pytest.MonkeyPatch, more: Record) -> None:
    Programs(monkeypatch, {"result": {"program_digest": D}})
    assert derived(options(program_digest=None), manifest(**more)) == (FACTS, DERIVED)


@pytest.mark.parametrize("listed", NO_BLOCK)
@pytest.mark.parametrize("components", NO_BLOCK)
def test_a_manifest_that_names_no_component_adds_no_digest_of_record(monkeypatch: pytest.MonkeyPatch, listed: object,
                                                                    components: object) -> None:
    Programs(monkeypatch, {"result": {"program_digest": D}})
    required = with_(with_(manifest(), "component_digests", listed if listed != {} else []), "components", components)
    assert derived(options(), required) == (FACTS, DERIVED), "the Commit's own digest is of record"
    assert derived(options(program_digest=None), required) == (
        None, {"used": False, "why": "Program(s) ['r0', 'r5'] not of record (of-record [])"})


WANT: Record = {"program_digest": D, "workload_sha256": "f" * 64}


def plan(tmp_path: Path, **fields: object) -> str:
    """A plan document on disk, with each of `fields` left out (`ABSENT`) or set."""
    doc: Record = {"schema": CP.SCHEMA, "key": dict(WANT), "utc": "2026-01-01T00:00:00Z", "wall_s": 1.5,
                   "producer_facts": {"facts": [["op", "m0", {"operand": "x"}]], "record": {"used": True, "members": 1}}}
    for field, value in fields.items():
        doc = with_(doc, field, value)
    (tmp_path / "plan.json").write_text(json.dumps(doc))
    return str(tmp_path / "plan.json")


@pytest.fixture
def keyed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Commit's key stood in: the real one hashes the workload, the manifest and the tree."""
    monkeypatch.setattr(CP, "key", lambda args: dict(WANT))


@pytest.mark.usefixtures("keyed")
def test_a_plan_under_the_commit_s_key_gives_its_facts_and_their_record(tmp_path: Path) -> None:
    got, record = CP.use(plan(tmp_path), None)
    assert got == ({("op", "m0"): {"operand": "x"}}, {"used": True, "members": 1})
    assert {k: v for k, v in record.items() if k != "check_s"} == {
        "path": str(tmp_path / "plan.json"), "schema": CP.SCHEMA, "key": WANT, "planned_utc": "2026-01-01T00:00:00Z", "plan_wall_s": 1.5,
        "used": True, "plan_recomputed": False, "members": 1}


@pytest.mark.usefixtures("keyed")
@pytest.mark.parametrize("no_key", [*NO_BLOCK, "", 0, []])
def test_a_plan_with_no_key_differs_from_the_commit_s_in_every_field(tmp_path: Path, no_key: object) -> None:
    got, record = CP.use(plan(tmp_path, key=no_key), None)
    assert got is None and (record["used"], record["plan_recomputed"], record["key"]) == (False, True, WANT)
    assert record["why"] == ("plan key differs from this Commit's: program_digest None != dddddddddddddddd, "
                             "workload_sha256 None != ffffffffffffffff")


@pytest.mark.usefixtures("keyed")
@pytest.mark.parametrize("facts", [*NO_BLOCK, {"facts": None}, {"facts": None, "record": None}, {"record": {}}])
def test_a_plan_that_holds_no_producer_facts_gives_the_commit_none_and_an_empty_record(tmp_path: Path, facts: object) -> None:
    got, record = CP.use(plan(tmp_path, producer_facts=facts), None)
    assert got == (None, {}) and (record["used"], record["plan_recomputed"], record["members"]) == (True, False, 0)


@pytest.mark.usefixtures("keyed")
def test_a_plan_s_facts_with_no_record_of_them_come_with_an_empty_one(tmp_path: Path) -> None:
    got, record = CP.use(plan(tmp_path, producer_facts={"facts": [["op", "m0", {}]]}), None)
    assert got == ({("op", "m0"): {}}, {}) and record["members"] == 1
