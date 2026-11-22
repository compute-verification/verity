"""The capture plan's rows and classes (`check.capture_plan`)."""
from __future__ import annotations

import gzip
import json
from typing import Any

import pytest
from verity_vllm.check import capture_plan as CP
from verity_vllm.check.replay.index import SAMPLER_OP


class _Index:
    """A `ProgramIndex` of described rows: each row's spec, its ``describe`` and its members' query families."""

    def __init__(self, rows: list[dict[str, Any]], families: list[dict[str, str]], embed: str | None = "model.embed_tokens",
                 checks: dict[str, Any] | None = None) -> None:
        self.described, self.families, self.embed = rows, families, embed
        self.rows = [{"i": i, "spec": d["spec"]} for i, d in enumerate(rows)]
        self.inst: dict[str, Any] = {} if checks is None else {"checks": checks}

    def describe(self, r):
        d = self.described[r["i"]]
        return None if d.get("unmodelled") else {k: v for k, v in d.items() if k != "spec"}

    def value_families(self, i):
        return dict(self.families[i])

    def embed_op_path(self):
        return self.embed


def _row(op: str, members: list[tuple[str, int, str]], family: str = "Gemm_v1", step: int = 0, row: int = 0) -> dict[str, Any]:
    return {"spec": f"{family}{{}}", "family": family, "statics": {}, "module": op.split("/")[0], "op_path": op, "members": members,
            "row": row, "step": step}


def _llama_like() -> _Index:
    rows = [_row("model.embed_tokens", [("out", 2048, "bf16")], "Embedding_v1"),
            _row("model.layers.0.mlp.down_proj/row_quant_3", [("q", 8192, "i8"), ("scale", 1, "f32")], "Int8RowQuant_v5"),
            _row("model.layers.0.mlp.down_proj", [("out", 2048, "bf16")]),
            _row("model.layers.0.mlp.down_proj/row_quant_67", [("q", 8192, "i8"), ("scale", 1, "f32")], "Int8RowQuant_v5", step=1),
            _row(SAMPLER_OP, [("sampled_token_ids", 1, "i64")], "TokenSelect_v1")]
    bounds = {"q": "call_boundaries", "scale": "call_boundaries"}
    fams = [{"out": "instance_outputs"}, bounds, {"out": "instance_outputs"}, bounds, {}]
    return _Index(rows, fams)


def test_each_row_names_each_members_capture_class():
    rows = CP.captured(_llama_like(), "r")
    bound = {"q": CP.CALL_BOUNDARY, "scale": CP.CALL_BOUNDARY}
    assert [r["classes"] for r in rows] == [{"out": CP.MODULE_OUTPUT}, bound, {"out": CP.MODULE_OUTPUT}, bound,
                                            {"sampled_token_ids": CP.SAMPLED}]
    assert rows[1]["op_path"] == "model.layers.0.mlp.down_proj/row_quant_3"
    assert {k for r in rows for k in r} == {"spec", "family", "statics", "module", "op_path", "members", "row", "step", "classes"}
    assert set(CP.CLASS_OF_FAMILY.values()) | {CP.SAMPLED, CP.INPUT_IDS} == set(CP.CLASSES)


@pytest.mark.parametrize("family,why", [("tp_rank_partials", "multi-GPU model is not served"),
                                        ("moe_block_stream", "no application serves MoE"),
                                        ("fa2_hidden_m1_stream", "has no class for")])
def test_a_value_the_plan_does_not_capture_is_refused_by_name(family, why):
    pi = _Index([_row("model.layers.0.mlp", [("out", 8, "bf16")])], [{"out": family}])
    with pytest.raises(ValueError, match=rf"r: row 0 .*model\.layers\.0\.mlp/out.*{why}"):
        CP.captured(pi, "r")


def test_a_check_row_is_left_out_and_any_other_row_without_members_refused():
    rows = [_row("m", [("out", 1, "i32")]), {**_row("c", []), "unmodelled": True}]
    assert [r["op_path"] for r in CP.captured(_Index(rows, [{"out": "instance_outputs"}, {}], checks={"of": ["len"], "at": [[1, 0, 1]]}))] == ["m"]
    with pytest.raises(ValueError, match=r"row 1 \(Gemm_v1\{\}\) has no committed member"):
        CP.captured(_Index(rows, [{"out": "instance_outputs"}, {}]), "r")


def _build(tmp_path, n: int = 2):
    for k in range(n):
        d = tmp_path / f"build_request_LP{k + 1}_T1"
        d.mkdir()
        (d / "instances.json.gz").write_bytes(gzip.compress(json.dumps({"k": k}).encode()))
    return tmp_path


def test_the_builds_plan_is_each_value_once_over_its_programs_with_the_prompt_ids(tmp_path):
    """A call boundary is one value at every count of its op over the request (`row_quant_3`, `row_quant_67`)."""
    seen = []

    def index(inst, program_dir):
        seen.append((inst, program_dir.rsplit("/", 1)[-1]))
        return _llama_like()

    plan = CP.of_build(_build(tmp_path), index=index)
    assert seen == [({"k": 0}, "build_request_LP1_T1"), ({"k": 1}, "build_request_LP2_T1")]
    assert plan == [
        {"op_path": "model.embed_tokens", "member": "in", "capture": CP.INPUT_IDS, "width": 1, "dtype": "i32", "definitions": []},
        {"op_path": "model.embed_tokens", "member": "out", "capture": CP.MODULE_OUTPUT, "width": 2048, "dtype": "bf16",
         "definitions": ["Embedding_v1"]},
        {"op_path": "model.layers.0.mlp.down_proj", "member": "out", "capture": CP.MODULE_OUTPUT, "width": 2048, "dtype": "bf16",
         "definitions": ["Gemm_v1"]},
        {"op_path": "model.layers.0.mlp.down_proj/row_quant", "member": "q", "capture": CP.CALL_BOUNDARY, "width": 8192, "dtype": "i8",
         "definitions": ["Int8RowQuant_v5"]},
        {"op_path": "model.layers.0.mlp.down_proj/row_quant", "member": "scale", "capture": CP.CALL_BOUNDARY, "width": 1,
         "dtype": "f32", "definitions": ["Int8RowQuant_v5"]},
        {"op_path": SAMPLER_OP, "member": "sampled_token_ids", "capture": CP.SAMPLED, "width": 1, "dtype": "i64",
         "definitions": ["TokenSelect_v1"]}]


def test_two_calls_of_one_op_at_one_step_are_refused_by_the_class_name_they_fold_to(tmp_path):
    def index(inst, program_dir):
        pi = _llama_like()
        pi.described[3]["step"] = 0
        return pi

    with pytest.raises(ValueError, match=r"row_quant_3 and model.layers.0.mlp.down_proj/row_quant_67 are two calls of one op at "
                                         r"step 0, and both fold to the class model.layers.0.mlp.down_proj/row_quant"):
        CP.of_build(_build(tmp_path), index=index)


def test_a_steps_rows_are_each_op_path_once_with_every_members_of_its_rows(tmp_path):
    """RoPE's query and key are rows of their own at every request row: one engine step commits the op path once, its
    members in the order its rows first give them."""
    rope = "model.layers.0.self_attn.rotary_emb"
    rows = [_row(rope, [("query", 2048, "bf16")], "RoPE_v1"), _row(rope, [("key", 512, "bf16")], "RoPE_v1"),
            _row(rope, [("query", 2048, "bf16")], "RoPE_v1", row=1), _row(rope, [("key", 512, "bf16")], "RoPE_v1", row=1)]
    got = CP.rows_of_build(_build(tmp_path), index=lambda *_: _Index(rows, [{"query": "instance_outputs", "key": "instance_outputs"}] * 4))
    assert [(r["op_path"], r["members"]) for r in got] == [(rope, [("query", 2048, "bf16"), ("key", 512, "bf16")])]


def test_a_value_two_rows_capture_differently_and_a_build_without_programs_are_refused(tmp_path):
    with pytest.raises(ValueError, match="no request Program"):
        CP.of_build(tmp_path, index=lambda *_: _llama_like())
    build = _build(tmp_path)
    twice = [_row("m", [("out", 8, "bf16")]), _row("m", [("out", 16, "bf16")], step=1)]
    with pytest.raises(ValueError, match=r"build_request_LP1_T1: m/out is module_output 16 bf16 here and module_output 8 bf16"):
        CP.of_build(build, index=lambda *_: _Index(twice, [{"out": "instance_outputs"}] * 2))
    with pytest.raises(ValueError, match="build_request_LP1_T1: no row reads the prompt ids"):
        CP.of_build(build, index=lambda *_: _Index(twice[:1], [{"out": "instance_outputs"}], embed=None))
