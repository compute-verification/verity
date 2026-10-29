"""`checks/step_segmentation.py` on synthetic inputs (laptop-safe, no fixture, no torch) and the carry-over rule of `lift_expected`.

The lift is the contract: a v1-shaped document of a one-token-per-step request whose segmentation rule missed its sampling family (one
segment, `component_steps_equal_cap: false`) and a segmented one differ exactly on steps / step_nodes / the chronology facts, never on
the component's identity or its sampling-event count.  `recompute` over a flat two-step request body counts both sampling families
and, for the greedy select, cuts one segment per event.
"""
from __future__ import annotations

import tomllib

import pytest

from tests.regression import lift_expected as LE
from tests.regression import resolver
from tests.regression.checks import step_segmentation as SS

CAP = 4
N = 40
GUMBEL = "GumbelTopPTokenSelect_v1{V=128256}"


def _docs(step_nodes: list[list[int]]) -> tuple[dict, dict]:
    wp = {"requests": [{"request_id": "r0", "index": 0, "prompt_len": 8, "cap": CAP, "arrive_step": 0,
                        "component": {"program_digest": "d" * 64, "n_nodes": N, "steps": len(step_nodes), "step_nodes": step_nodes}}],
          "inputs": {"derived": {f"8,{CAP - 1}": "/workspace/cp/sweep/row/build_request"}}}
    members = [[{"request_id": "r0", "s": s, "nodes": step_nodes[s] if s < len(step_nodes) else None}] for s in range(CAP)]
    chron = {"engine_steps": [{"e": s, "members": m} for s, m in enumerate(members)], "component_steps_equal_cap": len(step_nodes) == CAP}
    return wp, chron


HIST = {"build_request": {GUMBEL: CAP, "Gemm_v1{M=8}": 64, "TokenSelect_v1{V=5}": 0}}


def test_sampling_events_are_the_sampling_families_only():
    assert SS.sampling_events({GUMBEL: 4, "TokenSelect_v1{V=5}": 2, "GumbelTokenSelect_v1{V=5}": 1, "Gemm_v1{M=8}": 99}) == 7


def test_collapsed_and_segmented_documents_differ_on_the_segmentation_only():
    collapsed = SS.lift(*_docs([[0, N]]), HIST)
    segmented = SS.lift(*_docs([[0, 16], [16, 24], [24, 32], [32, N]]), HIST)
    r = collapsed["requests"]["r0"]
    assert (r["component"], r["steps"], r["sampling_events"]) == ("build_request", 1, CAP)
    assert collapsed["component_steps_equal_cap"] is False and collapsed["unsegmented_members"] == CAP - 1
    assert segmented["component_steps_equal_cap"] is True and segmented["unsegmented_members"] == 0
    d = resolver.diff(collapsed, segmented)
    assert sorted(x.split(":")[0] for x in d) == ["$.component_steps_equal_cap", "$.requests.r0.step_nodes_sha256", "$.requests.r0.steps",
                                                   "$.unsegmented_members"], d
    assert SS._consistent(segmented["requests"], {"build_request": True}) == {
        "r0": {"steps_eq_cap": True, "steps_eq_sampling_events": True, "segments_end_at_sampling_event": True}}
    assert SS._consistent(collapsed["requests"], {})["r0"]["steps_eq_sampling_events"] is False


def test_a_component_without_a_histogram_states_no_sampling_count():
    assert "sampling_events" not in SS.lift(*_docs([[0, N]]), {})["requests"]["r0"]


def _flat_request(family: str):
    """A flat two-step request body (what derive_step emits: the selects at the root body's top level, not inside a call):
    step 0 = [select(l0)], step 1 = [F32Add(temp, temp), select(l1)]."""
    from verity.ml.circuits.codec import encode_program
    from verity.ml.circuits.defs import CompositeDefinition, bind
    from verity.ml.circuits.program import Program
    from verity.ml.circuits.refs import tuple_of
    from verity.ml.circuits.types import Array, Tuple, Value
    from verity.ml.definitions.registry import b1, sampling
    from verity.ml.definitions.registry import prims as P
    from verity.ml.definitions.registry.prims import BF16, F32, I32
    V, I64 = 8, Value(64)
    greedy = family == "TokenSelect_v1"

    def sel(B, logits, seed, pos, temp):
        return B.call(bind(b1.TokenSelect, V=V), logits) if greedy else B.call(bind(sampling.GumbelTokenSelect, V=V), logits, seed, pos, temp)

    def body(B, S, l0, l1, seed, pos, temp):
        t0 = sel(B, l0, seed, pos, temp)
        t2 = B.call(P.F32Add, temp, temp)
        return tuple_of(t0, sel(B, l1, seed, pos, t2))
    sig = ((("l0", Array(V, BF16)), ("l1", Array(V, BF16)), ("seed", I64), ("pos", I64), ("temp", F32)), Tuple(I32, I32))
    req = CompositeDefinition(f"SegFixture{family.split('_')[0]}", 1, (), lambda S: sig, body, doc="step_segmentation fixture",
                              conformance="model-untested", register=False)
    return encode_program(Program(bind(req)))


@pytest.mark.parametrize("family", ["TokenSelect_v1", "GumbelTokenSelect_v1"])
def test_recompute_counts_the_sampling_events_of_a_flat_request(family):
    got = SS.recompute(_flat_request(family))
    assert got["n_nodes"] == 3 and got["sampling_events"] == 2
    assert got["step_nodes"][0][0] == 0 and got["step_nodes"][-1][1] == 3 and got["segments_end_at_sampling_event"] is True
    if family == "TokenSelect_v1":                     # every segmentation rule names the greedy select: one segment per event
        assert got["step_nodes"] == [[0, 1], [1, 3]]
    # the Gumbel body's `steps` is the rule under test (request_component's): 2 when the rule names GumbelTokenSelect_v1, 1 when not


def test_rows_the_index_does_not_name_are_carried_over(tmp_path):
    carried = {"old": {"row": 11, "class": "GREEN", "selection": "green", "tp": 1, "batch": 1, "oracle": "/o/old", "programs_state": "uploaded",
                       "local": {}, "artifacts": {"records": "art:" + "a" * 64, "programs": ""},
                       "files": {"row.log": {"role": "record", "bytes": 3, "sha256": "b" * 64}}}}
    index = {"rows": {"new": {"row": 101, "class": "GREEN", "selection": "green", "dir": "/o/new",
                              "files": {"stages.txt": {"bytes": 5, "sha256": "c" * 64, "extracted": True}}}}}
    text, ftext, rows_out = LE.build_fixtures(index, tmp_path, {"record"}, carried=carried)
    fx, files = tomllib.loads(text), tomllib.loads(ftext)
    assert list(fx["rows"]) == ["old", "new"] and list(rows_out) == ["old", "new"] and "files" not in fx["rows"]["old"]
    assert fx["rows"]["old"]["artifacts"]["records"] == "art:" + "a" * 64 and files["rows"]["old"]["files"]["row.log"]["sha256"] == "b" * 64
    assert fx["rows"]["new"]["selection"] == "green" and fx["rows"]["new"]["artifacts"] == {"records": "", "programs": ""}
    assert list(files["rows"]["new"]["files"]) == ["stages.txt"]
    assert LE.build_fixtures(index, tmp_path, {"record"}, carried={"new": dict(carried["old"], row=101)})[2]["new"]["oracle"] == "/o/new"


@pytest.mark.slow
def test_append_keeps_the_existing_text_and_refuses_a_known_key(tmp_path, monkeypatch):
    import json
    fx, files = tmp_path / "fixtures.toml", tmp_path / "files.toml"
    head = ('schema = "verity-vllm/r19-reference-fixtures/v1"\nsnapshot_artifact = "art:' + "e" * 64 + '"\n\n[artifacts]\n'
            'commit_logs = "art:' + "f" * 64 + '"   # a hand comment\n\n[rows."old"]\nrow = 11\nclass = "GREEN"\n')
    fhead = '[rows."old".files."a.json"]\nrole = "record"\nbytes = 2\n'
    fx.write_text(head)
    files.write_text(fhead)
    monkeypatch.setattr(resolver, "FIXTURES", fx)
    monkeypatch.setattr(resolver, "FILES", files)
    monkeypatch.setattr(resolver, "EXPECTED_DIR", tmp_path / "expected")
    row = tmp_path / "oracle" / "new"
    row.mkdir(parents=True)
    (row / "stages.txt").write_text("done\n")
    idx = tmp_path / "index.json"
    idx.write_text(json.dumps({"rows": {"new": {"row": 101, "class": "GREEN", "selection": "green", "dir": str(row),
                                                "files": {"stages.txt": {"bytes": 5, "sha256": "c" * 64, "extracted": True}}}}}))
    argv = ["--index", str(idx), "--oracle", str(tmp_path / "oracle"), "--append"]
    assert LE.main(argv) == 0
    text = fx.read_text()
    assert text.startswith(head.rstrip("\n") + "\n\n[rows.\"new\"]\n")
    assert list(tomllib.loads(text)["rows"]) == ["old", "new"]
    assert files.read_text().startswith(fhead + "\n[rows.\"new\".files.\"stages.txt\"]\n")
    assert list(tomllib.loads(files.read_text())["rows"]) == ["old", "new"]
    assert json.loads((tmp_path / "expected" / "new.json").read_text())["row"] == 101
    assert not (tmp_path / "expected" / "old.json").exists()
    assert LE.main(argv) == 2 and fx.read_text() == text
