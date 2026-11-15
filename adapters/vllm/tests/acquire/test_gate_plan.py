"""`verity-vllm gate plan` (`verity_vllm.acquire.gate`) through the command line, on a synthetic manifest and runtime tree: the
plan it writes, what it says of it, its exit, the request Programs it reads under a row dir, and its comparison with another plan."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from verity_vllm.acquire import gate as G
from verity_vllm.correspondence import runtime as RC
from verity_vllm.correspondence.reader_for_acquire import (
    RuntimeTree,
    runtime_tree_from_paths,
)
from verity_vllm.pipeline import cli
from verity_vllm.query.manifest import format as RM

PD = hashlib.sha256(b"program").hexdigest()
TREE = {"": ["Wrap", "Module"], "body.embed": ["Embed", "Module"], "body.norm": ["Norm", "Module"], "logits": ["Logits", "Module"]}


def row(family: str, op: str, member: str, **extra: Any) -> dict[str, Any]:
    return {"family": family, "program_digest": PD, "request_id": "r0", "step": 0, "invocation": 0, "op_path": op,
            "output_member": member, "element_range": [0, 8], "rank": 0, **extra}


ROWS = [row("input_token_ids", "body.embed", "in", rows=3, width=1, dtype="i32"),
        row("instance_outputs", "body.embed", "out", spec="Embedding_v1"),
        row("instance_outputs", "body.norm", "out", spec="Norm_v1"),
        row("instance_outputs", "logits", "out", spec="Gemm_v1")]
#: a Value of a module the runtime tree does not have
ABSENT = row("instance_outputs", "body.gone", "out", spec="Norm_v1")


def manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fams: dict[str, int] = {}
    for r in rows:
        fams[r["family"]] = fams.get(r["family"], 0) + 1
    return {"schema": RM.SCHEMA, "program_digest": PD, "identities": rows, "populations": {"identities": len(rows), "by_family": fams},
            "wrapper_prefix": "", "op_path_aliases": {}, "manifest_digest": RM.manifest_digest(rows)}


def tree() -> RuntimeTree:
    return runtime_tree_from_paths(TREE, runner={"sample": True, "sampler_module": True, "sampler_class": ["Sampler"]})


def gate_plan(d: Path, rows: list[dict[str, Any]], *more: str) -> tuple[int, dict[str, Any]]:
    """`gate plan` of ``rows`` into ``d``: its exit and the plan it wrote."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(manifest(rows)))
    tree().save(d / "runtime_tree.json")
    rc = cli.main(["gate", "plan", "--manifest", str(d / "manifest.json"), "--runtime", str(d / "runtime_tree.json"), "--out", str(d), *more])
    return rc, json.loads((d / "acquisition_plan.json").read_text())


def test_a_complete_plan_is_written_and_named(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc, doc = gate_plan(tmp_path, ROWS, "--row-id", "row-1")
    want = G.run(manifest(ROWS), tree())
    assert rc == 0 and want.complete and doc == want.to_json()
    first, second = capsys.readouterr().out.splitlines()
    assert first == (f"[gate] row-1 plan {want.digest[:16]} values {len(want.values)} (unsupported 0) sites {len(want.chosen_sites())} "
                     f"hooks {len(want.hooks())} residuals {len(want.residuals)}")
    assert second == ("[gate]   correspondence source=v1_annotations programs=0 absent=0 module_paths=0 modules confirmed=0 "
                      "unconfirmed=0 problems=0")


def test_a_value_without_a_site_is_named_and_the_exit_is_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc, doc = gate_plan(tmp_path, [*ROWS, ABSENT])
    want = G.run(manifest([*ROWS, ABSENT]), tree())
    assert rc == 1 and [v.value_id for v in want.unsupported] == ["body.gone/out"] and doc == want.to_json()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith(f"[gate]  plan {want.digest[:16]} values {len(want.values)} (unsupported 1) ")
    assert lines[2:] == [f"[gate]   UNSUPPORTED body.gone/out: {want.unsupported[0].why[:160]}"]


def test_the_request_programs_under_a_row_dir_are_read(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A row dir's `build_request*/` with a `result.json` are the Programs; one that records no correspondence digest is
    listed absent and the plan reads the manifest's annotations, as it does with no Program at all."""
    row_dir = tmp_path / "row"
    for name in ("build_request", "build_request_b"):
        (row_dir / name).mkdir(parents=True)
        (row_dir / name / "result.json").write_text(json.dumps({"program": {}}))
    (row_dir / "build_request_c").mkdir()
    rc, doc = gate_plan(tmp_path / "out", ROWS, "--programs-root", str(row_dir), "--program", str(row_dir / "build_request_c"))
    assert rc == 0 and doc["correspondence"]["source"] == "v1_annotations"
    assert doc["correspondence"]["absent"] == ["build_request_c", "build_request", "build_request_b"]
    assert doc["plan_digest"] == G.run(manifest(ROWS), tree()).digest
    assert " programs=0 absent=3 " in capsys.readouterr().out.splitlines()[1]


def record(modules: list[str]) -> dict[str, Any]:
    """A `runtime-correspondence/v1` document that owns one Call under each of ``modules``."""
    occs: list[dict[str, Any]] = [
        {"node": f"n{k}", "module": m, "module_class": None, "op": "aten.op", "rule": None, "outputs": [{"shape": [8], "dtype": "bf16"}],
         "written": [], "inputs": [], "lifetime": {}, "collective": None} for k, m in enumerate(modules)]
    calls = [{"occ": k, "member": None, "row": k, "hint": None, "returns": [{"port": 0, "output": 0, "leaves": [0, 8]}], "args": []}
             for k in range(len(modules))]
    return {"schema": RC.SCHEMA, "body": "program_fn", "inputs": ["ids"], "weights": None, "rank": None, "occurrences": occs, "calls": calls}


def test_a_record_that_owns_no_call_under_a_values_module_is_said(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A request Program with a correspondence record: the plan reads it, names each Value whose module the record owns no
    Call under, and says when the record is not the one the Build's `result.json` recorded."""
    doc = record(["body.embed", "logits"])
    d = tmp_path / "row" / "build_request"
    d.mkdir(parents=True)
    (d / "descriptor.json").write_text(json.dumps({"schema": "x", "program_fn": "f", "annotations": {"correspondence": doc}}))
    (d / "result.json").write_text(json.dumps({"program": {"correspondence_digest": "0" * 64}}))
    rc, plan = gate_plan(tmp_path / "out", ROWS, "--program", str(d))
    cp = plan["correspondence"]
    assert rc == 0 and cp["source"] == "correspondence" and cp["programs"] == {"build_request": RC.correspondence_digest(doc)}
    assert [(u["value"], u["module"]) for u in cp["modules"]["unconfirmed"]] == [("body.norm/out", "body.norm")]
    assert plan["plan_digest"] == G.run(manifest(ROWS), tree()).digest, "the record is provenance: the plan is the annotations' plan"
    lines = capsys.readouterr().out.splitlines()
    assert lines[1] == ("[gate]   correspondence source=correspondence programs=1 absent=0 module_paths=2 modules confirmed=3 "
                        "unconfirmed=1 problems=1")
    assert lines[2] == "[gate]   UNCONFIRMED body.norm/out (owning module body.norm): not a module the correspondence record owns Calls under"
    assert lines[3] == f"[gate]   correspondence problem: {cp['problems'][0][:200]}" and "0" * 16 in lines[3]


def test_against_compares_with_another_plan_value_by_value(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc, doc = gate_plan(tmp_path / "a", ROWS)
    other = tmp_path / "a" / "acquisition_plan.json"
    assert rc == 0
    rc, same = gate_plan(tmp_path / "b", ROWS, "--against", str(other))
    cmp = json.loads((tmp_path / "b" / "plan_compare.json").read_text())
    assert rc == 0 and same == doc and cmp == G.compare_plans(doc, doc) and cmp["identical"]
    capsys.readouterr()
    rc, fewer = gate_plan(tmp_path / "c", ROWS[:-1], "--against", str(other))
    cmp = json.loads((tmp_path / "c" / "plan_compare.json").read_text())
    assert rc == 1 and fewer["complete"] and cmp == G.compare_plans(fewer, doc)
    assert not cmp["identical"] and cmp["only_b"] == ["logits/out"] and not cmp["differing"]
    assert capsys.readouterr().out.splitlines()[2] == (
        f"[gate]   vs {other}: digest_equal False (v1_annotations vs v1_annotations) common {len(fewer['values'])} only_this 0 "
        "only_other 1 differing 0")


def test_against_names_a_value_whose_chosen_site_differs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc, doc = gate_plan(tmp_path / "a", ROWS)
    moved = json.loads(json.dumps(doc))
    value = next(v for v in moved["values"] if v["value"] == "logits/out")
    value["chosen"]["slot"] = "0"
    (tmp_path / "moved.json").write_text(json.dumps(moved))
    capsys.readouterr()
    rc, _ = gate_plan(tmp_path / "b", ROWS, "--against", str(tmp_path / "moved.json"))
    cmp = json.loads((tmp_path / "b" / "plan_compare.json").read_text())
    assert rc == 1 and cmp["plan_digest_equal"] and [d["value"] for d in cmp["differing"]] == ["logits/out"]
    this = next(v for v in doc["values"] if v["value"] == "logits/out")["chosen"]
    assert capsys.readouterr().out.splitlines()[3] == (
        f"[gate]   DIFFERS logits/out: this {[this['module'], this['kind'], this['slot']]} other {[this['module'], this['kind'], '0']}")
