"""`verity_vllm.engine.served` on documents that leave a block out: `declared` on a workload with no ``engine`` block, or with one
whose ``env``, ``args`` or ``reasons`` is absent, null or empty; `of_summary` on a summary with no ``served`` record or no
``versions`` block."""
from __future__ import annotations

import pytest

from verity_vllm.engine import served as SV

COMPILED = {"execution": "compiled"}
ENGINE_BLOCKS: list[dict | None] = [None, {}, {"env": None, "args": None, "reasons": None}, {"env": {}, "args": {}, "reasons": {}}]


def test_a_workload_with_no_engine_block_declares_what_its_execution_and_the_harness_set():
    d = SV.declared(COMPILED)
    assert {v["source"] for v in d.values()} == {"engine.env.ENGINE", "execution compiled"}
    assert "VLLM_BATCH_INVARIANT" not in d and "disable_custom_all_reduce" not in d


@pytest.mark.parametrize("engine", ENGINE_BLOCKS, ids=["null", "empty", "blocks null", "blocks empty"])
def test_an_engine_block_that_holds_nothing_declares_what_no_engine_block_does(engine):
    assert SV.declared({**COMPILED, "engine": engine}) == SV.declared(COMPILED)
    assert SV.declared({"engine": engine}) == SV.declared({})


def test_a_row_files_switch_without_a_reasons_block_has_no_reason():
    d = SV.declared({**COMPILED, "engine": {"env": {"VLLM_USE_DEEP_GEMM": "0"}, "args": {"enforce_eager": True}}})
    row = {"source": "row file", "reason": None}
    assert {k: v for k, v in d.items() if v["source"] == "row file"} == {
        "enforce_eager": {"value": True, **row}, "compilation_mode": {"value": "NONE", **row},
        "cudagraph_mode": {"value": "NONE", **row}, "VLLM_USE_DEEP_GEMM": {"value": False, **row}}


def test_batch_invariance_is_followed_by_vllms_own_switches_only_where_a_layer_turns_it_on():
    on = SV.declared({**COMPILED, "engine": {"env": {"VLLM_BATCH_INVARIANT": "1"}, "reasons": {"VLLM_BATCH_INVARIANT": "told"}}})
    assert on["VLLM_BATCH_INVARIANT"] == {"value": True, "source": "row file", "reason": "told"}
    assert {k: v["value"] for k, v in on.items() if v["source"] == "vLLM ParallelConfig"} == SV.BATCH_INVARIANT[0]
    off = SV.declared({**COMPILED, "engine": {"env": {"VLLM_BATCH_INVARIANT": "0"}}})
    assert off["VLLM_BATCH_INVARIANT"]["value"] is False and not [k for k, v in off.items() if v["source"] == "vLLM ParallelConfig"]
    assert SV.declared({})["VLLM_BATCH_INVARIANT"] == {"value": True, "source": "execution None", "reason": SV.EXECUTIONS[None][1]}
    assert SV.declared({})["disable_custom_all_reduce"]["source"] == "vLLM ParallelConfig"


@pytest.mark.parametrize("summary", [None, {}, {"versions": None}, {"versions": {}}, {"served": None}, {"served": {}},
                                     {"served": {}, "versions": None}, {"served": None, "versions": {"served": None}}])
def test_a_summary_without_a_served_record_has_none(summary):
    assert SV.of_summary(summary) is None


def test_a_summarys_own_served_record_is_read_before_its_versions():
    top, inner = {"ok": False}, {"ok": True}
    assert SV.of_summary({"versions": {"served": inner}}) is inner
    assert SV.of_summary({"served": None, "versions": {"served": inner}}) is inner
    assert SV.of_summary({"served": {}, "versions": {"served": inner}}) is inner
    assert SV.of_summary({"served": top, "versions": {"served": inner}}) is top
    assert SV.of_summary({"served": top, "versions": None}) is top
