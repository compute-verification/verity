"""`verity_vllm.pipeline.tp.rank_worker`: what a rank binds when its committed run ends
(`TP2CaptureWorkerExtension._tp2_binding`).  The rank fixes its binding map over its own commitment and holds it to a
population: with a required manifest, the manifest's rows for this rank; without one, the structure the committer saw over
the run's steps, with the member each collective site owes on this rank (the logits site its shard, every other site its
part).  A batch of two is bound by the rows the rank's collector recorded, and a rank that binds a request's rows to another
request's block fails by name.  A field the driver or the manifest did not state reads as none.  CPU-only: the committer is
the library's host committer over a stream the test names, and the manifest is a file the test writes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

from verity_vllm.commit import binding
from verity_vllm.commit.committer.native_host import NativeHostCommitter
from verity_vllm.engine.profiles import layer_classes as LC
from verity_vllm.pipeline.tp import rank_worker as RW
from verity_vllm.pipeline.tp.rank_worker import TP2CaptureWorkerExtension
from verity_vllm.query.manifest.format import SCHEMA, manifest_digest

PROGRAM = b"the rank's program"
PD = hashlib.sha256(PROGRAM).hexdigest()
RANK = 1
#: the two collective sites of the stand-in model, and the member each owes on rank 1
SITES = [("m.proj", LC.ROW_LINEAR), ("m.logits", LC.LOGITS)]
MEMBERS = [("m.proj", f"part_rank{RANK}"), ("m.logits", f"shard_rank{RANK}")]
STEPS = 3
ABSENT: Any = object()
#: a batch of two: the rows each request has at each engine step, and the ids the engine knows the requests by
DECLARED = {0: [("r0", 3), ("r1", 5)], 1: [("r0", 1), ("r1", 1)]}
BATCH_MEMBERS = ("model.embed_tokens/in", f"m.proj/part_rank{RANK}", "runner.sampler/sampled_token_ids")
ENGINE_ROWS = {0: [("id-a", 0, 3), ("id-b", 3, 8)], 1: [("id-a", 0, 1), ("id-b", 1, 2)]}


def _committed(stream: list[list[tuple[str, bytes]]]) -> tuple[Any, Any]:
    com = NativeHostCommitter(None, run_id="run", hash_threads=1, program_digest=hashlib.sha256(PROGRAM).digest())
    return com, com.commit_offline(stream)


def _stream(members: list[tuple[str, str]] = MEMBERS, steps: int = STEPS) -> list[list[tuple[str, bytes]]]:
    return [[(f"{op}/{member}", bytes([step + 1]) * (100 + 50 * i)) for i, (op, member) in enumerate(members)] for step in range(steps)]


def _rank(com: Any, **state: Any) -> TP2CaptureWorkerExtension:
    """A rank at the end of its run: its state as `tp2_commit_make` leaves it, less every field given as `ABSENT`."""
    rank = TP2CaptureWorkerExtension()
    given = {"program_digest": PD, "com": com, "rank": RANK, "world": 2, "tps": SimpleNamespace(sites=list(SITES)), **state}
    rank._tp2_commit = {k: v for k, v in given.items() if v is not ABSENT}
    return rank


def _bind(stream: list[list[tuple[str, bytes]]] | None = None, **state: Any) -> tuple[dict[str, Any], TP2CaptureWorkerExtension]:
    com, rc = _committed(_stream() if stream is None else stream)
    rank = _rank(com, **state)
    answer = rank._tp2_binding(rc)
    assert answer is not None
    return answer, rank


def _rows(rank: int = RANK, members: list[tuple[str, str]] = MEMBERS) -> list[dict[str, Any]]:
    return [{"family": "instance_outputs", "program_digest": PD, "request_id": "r0", "step": step, "invocation": 0, "op_path": op,
             "output_member": member, "element_range": [0, None], "rank": rank} for step in range(STEPS) for op, member in members]


def _manifest(tmp_path: Path, rows: list[dict[str, Any]], **fields: Any) -> tuple[str, dict[str, Any]]:
    doc = {"schema": SCHEMA, "program_digest": PD, "manifest_digest": manifest_digest(rows), "identities": rows,
           **{k: v for k, v in fields.items() if v is not ABSENT}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(doc))
    return str(path), doc


def _population(rank: TP2CaptureWorkerExtension) -> set[tuple[int, str, str]]:
    """(step, op_path, member) of every identity the rank held its map to."""
    return {(i[2], i[4], i[5]) for i in rank._tp2_commit["pop"]}


# ---- a rank with no Program

@pytest.mark.parametrize("digest", [ABSENT, None, ""], ids=["absent", "none", "empty"])
def test_a_rank_with_no_program_digest_says_so_and_binds_nothing(digest):
    answer, rank = _bind(program_digest=digest)
    assert list(answer) == ["skipped"] and "no program_digest for this rank" in answer["skipped"]
    assert "bmap" not in rank._tp2_commit and "pop" not in rank._tp2_commit


# ---- no manifest: the structure over the run's steps, with each collective site's member on this rank

def test_without_a_manifest_the_rank_is_held_to_its_structure_over_the_runs_steps():
    answer, rank = _bind()
    assert sorted(answer) == ["context", "coverage", "entries_n", "map_digest", "note", "population", "program_digest", "rank", "rules_version",
                              "structure_modules"]
    assert (answer["rank"], answer["program_digest"], answer["rules_version"]) == (RANK, PD, binding.RULES_VERSION)
    pop = answer["population"]
    assert pop["source"].startswith("structure(") and pop["manifest"] is None
    assert (pop["n"], pop["structural_n"], pop["steps"], pop["requests"], pop["ranks"]) == (6, 6, STEPS, ["r0"], [RANK])
    assert pop["tp_partial_members"] == 2 and pop["step_rows_recorded"] == 0
    assert answer["coverage"]["ok"] is True and (answer["coverage"]["expected_n"], answer["coverage"]["bound_n"]) == (6, 6)
    assert "composition" not in answer["coverage"]
    assert answer["context"] == {"ok": True, "why": "PASS"}
    assert answer["note"] == "population = structure x workload (no manifest given); B=1: request r0 = every step"
    # the map and the population stay with the rank: the sampled replay and the cross-rank picks read them
    assert rank._tp2_commit["bmap"]["digest"] == answer["map_digest"] and answer["entries_n"] == rank._tp2_commit["bmap"]["entries_n"] == 6
    assert _population(rank) == {(step, op, member) for step in range(STEPS) for op, member in MEMBERS}
    assert pop["digest"] == binding.population_digest(rank._tp2_commit["pop"])[0]


def test_the_logits_site_owes_its_shard_and_every_other_site_its_part():
    """A rank whose logits site committed a part where its shard belongs has not bound what the structure requires."""
    answer, rank = _bind(_stream([("m.proj", f"part_rank{RANK}"), ("m.logits", f"part_rank{RANK}")]))
    assert ("m.logits", f"shard_rank{RANK}") in {(op, member) for _step, op, member in _population(rank)}
    assert answer["coverage"]["ok"] is False
    assert {(m["op_path"], m["output_member"]) for m in answer["coverage"]["missing"]} == {("m.logits", f"shard_rank{RANK}")}


def test_a_site_the_rank_did_not_commit_fails_its_coverage():
    """The population is the structure's, not the collector's: a rank that committed one site of two is short the other."""
    answer, _rank_ = _bind(_stream(MEMBERS[:1]))
    assert answer["population"]["tp_partial_members"] == 2 and answer["population"]["n"] == 6
    assert answer["coverage"]["ok"] is False and answer["coverage"]["missing_n"] == STEPS
    assert {m["op_path"] for m in answer["coverage"]["missing"]} == {"m.logits"}


def test_a_site_is_required_on_the_rank_that_binds_it():
    com, rc = _committed(_stream([("m.proj", "part_rank0"), ("m.logits", "shard_rank0")]))
    rank = _rank(com, rank=0)
    answer = rank._tp2_binding(rc)
    assert answer is not None and answer["rank"] == 0 and answer["coverage"]["ok"] is True
    assert {(op, member) for _step, op, member in _population(rank)} == {("m.proj", "part_rank0"), ("m.logits", "shard_rank0")}


@pytest.mark.parametrize("world", [ABSENT, None, 1], ids=["absent", "none", "one"])
def test_a_world_of_one_owes_no_collective_member(world):
    """With no collective source and one rank, the structure gains no site: the host committer of this test saw no module,
    so nothing is required, where two ranks' sites make the two modules of the tests above."""
    answer, rank = _bind(tps=None, world=world)
    assert answer["population"]["tp_partial_members"] == 0 and answer["structure_modules"] == 0
    assert answer["population"]["n"] == answer["coverage"]["expected_n"] == 0 and _population(rank) == set()
    assert _bind()[0]["structure_modules"] == len(SITES)


# ---- the tables the driver states for each request

@pytest.mark.parametrize("stated", [ABSENT, None, {}], ids=["absent", "none", "empty"])
def test_a_rank_told_no_arrivals_holds_the_request_to_the_runs_own_steps(stated):
    answer, rank = _bind(arrivals=stated)
    assert answer["population"]["steps"] == STEPS and answer["population"]["n"] == 6 and answer["coverage"]["ok"] is True
    assert _population(rank) == {(step, op, member) for step in range(STEPS) for op, member in MEMBERS}


@pytest.mark.parametrize(("arrivals", "at"), [({"r0": 1}, (1, 2)), ({"r7": 1}, (0, 1)), (ABSENT, (0, 1))], ids=["arrived-at-1", "another-request", "absent"])
def test_a_requests_arrival_places_the_manifests_rows_and_leaves_the_runs_steps(tmp_path, arrivals, at):
    """A manifest's steps are the request's own.  A request that arrived at step 1 is held to the manifest's steps 0 and 1 at
    the run's steps 1 and 2; a request the table leaves out arrived at step 0.  The arrival alone does not cut the steps the
    run spans."""
    path, doc = _manifest(tmp_path, [row for row in _rows() if row["step"] < 2])
    answer, rank = _bind(required_manifest=path, manifest_doc=doc, arrivals=arrivals)
    assert answer["population"]["steps"] == STEPS and answer["population"]["n"] == 4
    assert _population(rank) == {(step, op, member) for step in at for op, member in MEMBERS}
    assert answer["coverage"]["missing_n"] == 0


@pytest.mark.parametrize(("stated", "requests", "table"), [
    (None, 2, {"r0": 0, "r1": 0}), ({}, 2, {"r0": 0, "r1": 0}), ({"r1": 4}, 2, {"r0": 0, "r1": 4}), ({"r0": "3", "r7": 9}, 1, {"r0": 3}), ({"r0": 3}, 0, {})],
    ids=["none", "empty", "one-left-out", "a-number-as-text", "no-request"])
def test_a_table_per_request_reads_zero_for_a_request_it_leaves_out(stated, requests, table):
    assert RW._per_request(stated, requests) == table


# ---- a required manifest: its rows for this rank

def test_with_a_manifest_the_rank_is_held_to_the_manifests_rows_for_it(tmp_path):
    path, doc = _manifest(tmp_path, _rows() + _rows(rank=0, members=[("m.proj", "part_rank0")]))
    answer, rank = _bind(required_manifest=path, manifest_doc=doc)
    pop = answer["population"]
    assert pop["source"].startswith("required-manifest") and (pop["n"], pop["structural_n"]) == (6, 6)
    assert (pop["manifest"]["rows_for_rank"], pop["manifest"]["rows_all_ranks"]) == (6, 9)
    assert pop["manifest"]["path"] == path and pop["manifest"]["manifest_digest_verified"] is True
    assert all(i[7] == RANK for i in rank._tp2_commit["pop"])
    assert answer["coverage"]["ok"] is True and answer["coverage"]["composition"]["ok"] is True
    assert answer["note"].startswith("population = required manifest rows with rank=r")


def test_a_manifest_row_the_rank_did_not_bind_fails_its_coverage(tmp_path):
    path, doc = _manifest(tmp_path, _rows(members=MEMBERS + [("m.other", f"part_rank{RANK}")]))
    answer, _rank_ = _bind(required_manifest=path, manifest_doc=doc)
    assert answer["coverage"]["ok"] is False and answer["coverage"]["missing_n"] == STEPS
    assert {m["op_path"] for m in answer["coverage"]["missing"]} == {"m.other"}


def test_a_manifest_with_no_row_for_the_rank_is_an_error_by_name(tmp_path):
    path, doc = _manifest(tmp_path, _rows(rank=0))
    answer, rank = _bind(required_manifest=path, manifest_doc=doc)
    assert answer == {"error": f"required manifest has no rows for rank {RANK} ({path})", "rank": RANK}
    assert "bmap" not in rank._tp2_commit


def test_a_manifest_that_cannot_be_read_is_an_error_with_its_trace(tmp_path):
    answer, _rank_ = _bind(required_manifest=str(tmp_path / "none.json"))
    assert sorted(answer) == ["error", "trace"] and "FileNotFoundError" in answer["error"] and "none.json" in answer["trace"]


@pytest.mark.parametrize("doc_field", [ABSENT, None, {}], ids=["absent", "none", "empty"])
def test_a_rank_with_no_manifest_document_states_no_program_of_the_manifests(tmp_path, doc_field):
    """The driver hands the manifest's document beside its path; a rank that was handed the path alone reads the file and
    states that the manifest names no Program for it."""
    path, _doc = _manifest(tmp_path, _rows())
    answer, _rank_ = _bind(required_manifest=path, manifest_doc=doc_field)
    assert answer["population"]["manifest"]["program_digest_for_rank"] is None and answer["coverage"]["ok"] is True


@pytest.mark.parametrize("per_rank", [ABSENT, None, {}, {"0": "ff" * 32}], ids=["absent", "none", "empty", "another-rank"])
def test_a_manifest_that_names_no_program_for_the_rank_does_not_fail_it(tmp_path, per_rank):
    path, doc = _manifest(tmp_path, _rows(), program_digest_per_rank=per_rank)
    answer, _rank_ = _bind(required_manifest=path, manifest_doc=doc)
    assert answer["population"]["manifest"]["program_digest_for_rank"] is None
    assert answer["coverage"]["ok"] is True and "error" not in answer["coverage"]


def test_the_manifests_program_for_the_rank_is_the_ranks_or_the_coverage_fails(tmp_path):
    path, doc = _manifest(tmp_path, _rows(), program_digest_per_rank={str(RANK): PD, "0": "ff" * 32})
    answer, _rank_ = _bind(required_manifest=path, manifest_doc=doc)
    assert answer["population"]["manifest"]["program_digest_for_rank"] == PD and answer["coverage"]["ok"] is True
    other = "ab" * 32
    path, doc = _manifest(tmp_path, _rows(), program_digest_per_rank={str(RANK): other, "0": PD})
    answer, _rank_ = _bind(required_manifest=path, manifest_doc=doc)
    assert answer["coverage"]["ok"] is False
    assert answer["coverage"]["error"] == f"manifest Program digest for rank {RANK} {other[:16]} != this rank's Program {PD[:16]}"
    assert answer["coverage"]["missing_n"] == 0


@pytest.mark.parametrize(("unmodelled", "families"), [(ABSENT, ""), (None, ""), ({}, ""), ({"moe": 2, "attn": 1}, "attn x1; moe x2")],
                         ids=["absent", "none", "empty", "given"])
def test_an_incomplete_manifest_is_no_population_and_names_the_families_it_leaves_out(tmp_path, unmodelled, families):
    path, doc = _manifest(tmp_path, _rows(), complete=False, unmodelled=unmodelled)
    answer, _rank_ = _bind(required_manifest=path, manifest_doc=doc)
    assert answer["coverage"]["ok"] is False
    assert answer["coverage"]["error"] == "manifest incomplete: unmodelled Program families " + families
    # an incomplete manifest is named before the manifest's Program for the rank is compared
    path, doc = _manifest(tmp_path, _rows(), complete=False, unmodelled=unmodelled, program_digest_per_rank={str(RANK): "ab" * 32})
    answer, _rank_ = _bind(required_manifest=path, manifest_doc=doc)
    assert answer["coverage"]["error"].startswith("manifest incomplete")


def test_a_header_that_states_no_families_names_none(tmp_path, monkeypatch):
    """The header `load_required_manifest` gives always states the families; a header without them reads the same."""
    real = binding.load_required_manifest

    def without_families(*args: Any, **kwargs: Any) -> tuple[list[tuple], dict[str, Any]]:
        pop, hdr = real(*args, **kwargs)
        return pop, {k: v for k, v in hdr.items() if k != "unmodelled"}

    monkeypatch.setattr(binding, "load_required_manifest", without_families)
    path, doc = _manifest(tmp_path, _rows(), complete=False, unmodelled={"moe": 2})
    answer, _rank_ = _bind(required_manifest=path, manifest_doc=doc)
    assert answer["coverage"]["error"] == "manifest incomplete: unmodelled Program families "


# ---- a batch of two: the rows the rank's collector recorded, and the composition the manifest declares

def _batch_stream() -> list[list[tuple[str, bytes]]]:
    return [[(name, bytes([e + 1]) * (2 if name.endswith("sampled_token_ids") else sum(n for _r, n in DECLARED[e]))) for name in BATCH_MEMBERS]
            for e in sorted(DECLARED)]


def _batch_manifest(tmp_path: Path) -> tuple[str, dict[str, Any]]:
    rows = []
    for e, members in sorted(DECLARED.items()):
        for rid, n in members:
            for name in BATCH_MEMBERS:
                op, member = name.split("/", 1)
                family = {"sampled_token_ids": "sampled_token_ids", "in": "input_token_ids"}.get(member, "instance_outputs")
                rows.append({"family": family, "program_digest": PD, "request_id": rid, "step": e, "engine_step": e, "invocation": 0, "op_path": op,
                             "output_member": member, "element_range": [0, 1 if family == "sampled_token_ids" else n], "rows": n, "width": 1,
                             "rank": RANK})
    return _manifest(
        tmp_path, rows, **{"global": True}, addressing="request-local", program_digest_kind="workload", complete=True, unmodelled={},
        requests_declared=[{"request_id": f"r{k}", "workload_request_id": f"uuid-{k}", "index": k, "prompt_len": (3, 5)[k], "max_tokens": 2,
                            "arrive_step": 0} for k in range(2)],
        engine_steps=[{"engine_step": e, "members": [{"request_id": rid, "step": e, "rows": n} for rid, n in members],
                       "rows": sum(n for _r, n in members), "n_seq": len(members)} for e, members in sorted(DECLARED.items())],
        composition_reference={"input_token_ids": {"op_path": "model.embed_tokens", "output_member": "in", "per_row": 1},
                               "sampled_token_ids": {"op_path": "runner.sampler", "output_member": "sampled_token_ids", "per_row": 1}})


def _bind_batch(tmp_path: Path, labels: dict[str, str], rows: Any = ENGINE_ROWS, **state: Any) -> dict[str, Any]:
    com, rc = _committed(_batch_stream())
    if rows is not ABSENT:
        com.step_rows = rows
    path, doc = _batch_manifest(tmp_path)
    rank = _rank(com, tps=SimpleNamespace(sites=SITES[:1]), n_requests=2, request_ids=labels, required_manifest=path, manifest_doc=doc, **state)
    answer = rank._tp2_binding(rc)
    assert answer is not None
    return answer


def test_a_batch_of_two_is_bound_by_the_rows_the_ranks_collector_recorded(tmp_path):
    answer = _bind_batch(tmp_path, {"id-a": "r0", "id-b": "r1"})
    pop = answer["population"]
    assert (pop["requests"], pop["n"], pop["step_rows_recorded"], pop["steps"]) == (["r0", "r1"], 12, 2, 2)
    assert answer["coverage"]["ok"] is True and answer["coverage"]["missing"] == []
    comp = answer["coverage"]["composition"]
    assert comp["ok"] is True and comp["checked_steps"] == 2 and comp["problems"] == []
    assert answer["note"].endswith("; B=2: request rows from the rank's collector step_rows")


def test_a_rank_that_binds_a_requests_rows_to_the_other_requests_block_fails_by_name(tmp_path):
    """The two requests' labels the other way round: every member is still bound, each to a block of the size the other request
    declares, so the rank's composition is not the manifest's and the rank says which step and member."""
    answer = _bind_batch(tmp_path, {"id-a": "r1", "id-b": "r0"})
    cov = answer["coverage"]
    assert cov["ok"] is False
    assert cov["error"].startswith(f"rank {RANK}: composition: engine step 0 composition on model.embed_tokens/in")
    assert "missing ['r0 x3', 'r1 x5']" in cov["error"] and "unexpected ['r0 x5', 'r1 x3']" in cov["error"]
    assert cov["composition"]["ok"] is False and cov["composition"]["problems"][0]["engine_step"] == 0
    assert isinstance(cov["missing"], list) and cov["missing_n"] == len(cov["missing"]) == 2
    assert {(m["request_id"], m["op_path"]) for m in cov["missing"]} == {("r1", "model.embed_tokens"), ("r1", "m.proj")}


@pytest.mark.parametrize("rows", [ABSENT, None, {}], ids=["absent", "none", "empty"])
def test_a_batch_whose_collector_recorded_no_rows_has_no_coverage_verdict(tmp_path, rows):
    answer = _bind_batch(tmp_path, {"id-a": "r0", "id-b": "r1"}, rows=rows)
    assert list(answer) == ["skipped"] and answer["skipped"].startswith("B=2 > 1 but the rank's collector recorded no step_rows")


@pytest.mark.parametrize("n_requests", [ABSENT, None, 1], ids=["absent", "none", "one"])
def test_a_rank_told_of_no_batch_binds_one_request(n_requests):
    answer, _rank_ = _bind(n_requests=n_requests)
    assert answer["population"]["requests"] == ["r0"] and answer["note"].endswith("; B=1: request r0 = every step")


# ---- what the composition check answers, in every shape

def _composed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, composition: dict[str, Any], coverage: dict[str, Any] | None = None) -> dict[str, Any]:
    """The rank's coverage where the binding module's composition check (and, when given, its coverage check) answers as stated."""
    monkeypatch.setattr(binding, "composition_check", lambda hdr, bmap: dict(composition))
    if coverage is not None:
        monkeypatch.setattr(binding, "coverage_check", lambda pop, bmap: dict(coverage))
    path, doc = _manifest(tmp_path, _rows())
    answer, _rank_ = _bind(required_manifest=path, manifest_doc=doc)
    return answer["coverage"]


@pytest.mark.parametrize(("problems", "kept"), [(ABSENT, []), (None, []), ([], []), (["p0"], ["p0"]), (["p0", "p1", "p2", "p3", "p4"], ["p0", "p1", "p2"])],
                         ids=["absent", "none", "empty", "one", "five"])
def test_the_coverage_carries_the_first_three_problems_of_the_composition(monkeypatch, tmp_path, problems, kept):
    stated = {"ok": True, "checked_steps": 4} | ({} if problems is ABSENT else {"problems": problems})
    cov = _composed(monkeypatch, tmp_path, stated)
    assert cov["composition"] == {"ok": True, "checked_steps": 4, "problems": kept}
    assert cov["ok"] is True and "error" not in cov


@pytest.mark.parametrize(("missing", "kept"), [(ABSENT, []), (None, []), ([], []), (("m0", "m1"), ["m0", "m1"])], ids=["absent", "none", "empty", "given"])
def test_a_failed_composition_fails_the_coverage_and_keeps_what_was_missing(monkeypatch, tmp_path, missing, kept):
    coverage = {"ok": True, "expected_n": 6} | ({} if missing is ABSENT else {"missing": missing})
    cov = _composed(monkeypatch, tmp_path, {"ok": False, "error": "composition: step 3"}, coverage)
    assert cov["ok"] is False and cov["error"] == f"rank {RANK}: composition: step 3"
    assert cov["missing"] == kept and cov["expected_n"] == 6


@pytest.mark.parametrize("error", [ABSENT, None, ""], ids=["absent", "none", "empty"])
def test_a_failed_composition_that_states_no_error_is_named_all_the_same(monkeypatch, tmp_path, error):
    stated = {"ok": False} | ({} if error is ABSENT else {"error": error})
    cov = _composed(monkeypatch, tmp_path, stated)
    assert cov["ok"] is False and cov["error"] == f"rank {RANK}: composition check failed"


def test_an_incomplete_manifest_is_named_before_a_failed_composition(monkeypatch, tmp_path):
    monkeypatch.setattr(binding, "composition_check", lambda hdr, bmap: {"ok": False, "error": "composition: step 3"})
    path, doc = _manifest(tmp_path, _rows(), complete=False, unmodelled={"moe": 2})
    answer, _rank_ = _bind(required_manifest=path, manifest_doc=doc)
    assert answer["coverage"]["error"] == "manifest incomplete: unmodelled Program families moe x2"
