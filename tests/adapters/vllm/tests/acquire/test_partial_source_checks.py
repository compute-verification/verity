"""The value checks of `verity_vllm.acquire.sources.partial_source` that no other CPU test enters: the re-execution half of
`TPPartialSource.compare` (the rank-local GEMM run again on the tapped module input), the capture's align witness of a step the oracle
compare never reached, and the late-read detector's refusal of a received output that is in no decoder layer.  The rank's members are
committed by the CPU committer of `tests.check.opened` and read through `oracle_compare.OpenedReader`, opened and verified against the run
root, as a rank worker reads them.  The late-read detector's verdicts are `tests/engine/test_tp2_t6_4_check.py`'s."""
from __future__ import annotations

import json
import types
from typing import Any

import pytest
import torch

from tests.check.opened import commit_steps
from verity_vllm.acquire.sources import oracle_witness as OW
from verity_vllm.acquire.sources import partial_source as P
from verity_vllm.check import oracle_compare as OC

OPROJ, MOE = "model.layers.0.self_attn.o_proj", "model.layers.0.mlp.experts"
PART = f"{OPROJ}/part_rank0"


class Gemm:
    """A layer's quantization method: `apply` is the rank-local GEMM on the shard, plus the bias it is handed."""

    def apply(self, layer: Any, x: Any, bias: Any = None) -> Any:
        y = x @ layer.weight.T
        return y if bias is None else y + bias


class RowParallelLinear:
    """Named as vLLM's layer: the source knows a row-parallel site by its class's name."""

    reduce_results, tp_size = True, 2

    def __init__(self, **flags: Any) -> None:
        self.weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        self.bias = torch.tensor([0.5, -1.0, 2.0])
        self.quant_method = Gemm()
        self.__dict__.update(flags)

    def parameters(self) -> Any:
        return iter(())


def _x() -> Any:
    return torch.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 0.5, 2.0]])


def _value_check(layer: RowParallelLinear, part: Any, *, site: str = OPROJ) -> dict[str, Any]:
    """The value check of a rank that committed `part` as o_proj's partial, tapped it as committed, and tapped `_x()` as the input of
    module `site`."""
    com, run = commit_steps({0: [(PART, P._bytes(part))]})
    src = P.TPPartialSource(com, types.SimpleNamespace(named_modules=lambda: [(OPROJ, layer)]), rank=0, world=2)
    assert src.reexec_sites == [OPROJ]
    src.ref, src.ref_in = {0: {PART: P._bytes(part)}}, {0: {PART: (site, _x())}}
    res = src.compare(OC.OpenedReader(com, run))
    assert (res["tensors"], res["tap_equal"], res["tap_mismatch_n"], res["missing_n"], res["reexec_due"]) == (1, 1, 0, 0, True)
    assert src.ref == {} and src.ref_in == {}
    return res


# --- the re-execution of the rank-local GEMM ---------------------------------------------------------------------------------------------------

def test_a_committed_partial_that_is_the_gemm_of_the_tapped_input_passes_the_re_execution():
    layer = RowParallelLinear()
    res = _value_check(layer, _x() @ layer.weight.T + layer.bias)
    assert res["verdict"] == "PASS" and (res["reexec_tensors"], res["reexec_equal"], res["reexec_mismatch_n"]) == (1, 1, 0)
    assert res["by_member"] == {"part": {"tensors": 1, "tap_equal": 1, "reexec": 1, "reexec_equal": 1}} and res["reexec_skipped"] == {}
    assert res["value_source"]["source"] == "opened" and res["value_source"]["failed"] == 0


@pytest.mark.parametrize("flags", [{"tp_rank": 1}, {"skip_bias_add": True}])
def test_the_re_execution_adds_the_bias_only_where_the_layer_does(flags):
    layer = RowParallelLinear(**flags)
    assert _value_check(layer, _x() @ layer.weight.T)["verdict"] == "PASS"
    res = _value_check(layer, _x() @ layer.weight.T + layer.bias)
    assert res["verdict"] == "FAIL" and res["reexec_mismatch"] == [(0, PART, 24)] and res["reexec_equal"] == 0


def test_a_committed_partial_that_is_not_the_gemm_of_the_tapped_input_fails_the_re_execution_by_name():
    layer = RowParallelLinear()
    res = _value_check(layer, _x() @ layer.weight.T + layer.bias + 1.0)        # the tap agrees with the commit; the GEMM does not
    assert res["verdict"] == "FAIL" and (res["reexec_tensors"], res["reexec_equal"], res["reexec_mismatch_n"]) == (1, 0, 1)
    assert res["reexec_mismatch"] == [(0, PART, 24)] and res["by_member"]["part"]["reexec_equal"] == 0


def test_a_site_that_cannot_be_re_executed_is_skipped_by_name_and_the_check_fails():
    layer = RowParallelLinear(input_is_parallel=False)
    res = _value_check(layer, _x() @ layer.weight.T + layer.bias)
    assert res["verdict"] == "FAIL" and res["reexec_tensors"] == 0
    assert res["reexec_skipped"] == {OPROJ: "RuntimeError('input_is_parallel=False: the module splits its input; re-exec skipped')"}
    gone = _value_check(RowParallelLinear(), _x(), site="model.layers.9.self_attn.o_proj")     # a tapped input of no module of the model
    assert gone["verdict"] == "FAIL" and list(gone["reexec_skipped"]) == ["model.layers.9.self_attn.o_proj"]
    assert "quant_method" in gone["reexec_skipped"]["model.layers.9.self_attn.o_proj"]


# --- an align witness of a step the oracle compare never reached ---------------------------------------------------------------------------------

def test_a_witness_the_capture_recorded_at_a_step_the_commit_never_reached_is_unpaired_by_name(tmp_path):
    part = _x()
    (tmp_path / "coll.rank0.jsonl").write_text(
        json.dumps({"rank": 0, "step": 0, "k": 0, "kind": "all_reduce", "module": OPROJ, "shape": list(part.shape)}) + "\n")
    torch.save({"step": 0, "part": [part], "out": []}, tmp_path / "coll.rank0.step0.pt")
    with (tmp_path / OW.witness_file(0)).open("w") as f:
        for m in OW.MEMBERS:
            f.write(json.dumps({"rank": 0, "step": 2, "k": 1, "module": MOE, "member": m, "sha256": OW.digest(b"words")}) + "\n")
    com, run = commit_steps({0: [(PART, P._bytes(part))]})
    src = P.TPPartialSource(com, types.SimpleNamespace(named_modules=lambda: [(OPROJ, RowParallelLinear())]), rank=0, world=2)
    res = src.compare_match_oracle(str(tmp_path), OC.OpenedReader(com, run))
    assert res["witnesses"]["unpaired"] == [{"step": 2, "module": MOE, "in_commit": False, "in_capture": True}]
    assert res["witness_counts"] == {"equal": 0, "differs": 0, "unpaired": 1} and res["verdict"] == "FAIL"
    assert (res["steps_compared"], res["tensors"], res["equal"], res["mismatch_n"], res["missing_n"]) == (1, 1, 1, 0, 0)


# --- the late-read detector ------------------------------------------------------------------------------------------------------------------

def test_a_received_output_of_no_decoder_layer_is_refused_by_name():
    com, run = commit_steps({0: [("model.mlp.experts/out", b"a" * 8), ("model.norm/0", b"b" * 8)]})
    with pytest.raises(ValueError, match=r"^model\.mlp\.experts/out: a fused-MoE received output of no decoder layer"):
        P.late_read_check(com, run, 0)


def test_the_detectors_record_carries_its_schema_id_and_its_rank():
    com, run = commit_steps({0: [("model.norm/0", b"b" * 8)]})                 # a dense rank: no received output to compare
    res = P.late_read_check(com, run, 1)
    assert (res["schema"], res["rank"], res["applicable"], res["result"]) == ("verity/commit-integ/t6_4-check/v1", 1, False, None)
