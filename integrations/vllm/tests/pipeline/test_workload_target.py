"""The sweep workload generator's `--target`: a TP row's `env_required` is the declared target's environment, not the batch-invariant default."""
from __future__ import annotations

import json

import pytest

from verity_vllm.pipeline import workload as W


@pytest.mark.parametrize("row_id,target,env", [
    ("x__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager",
     {"compute_capability": [9, 0], "num_sms": 132, "batch_invariant": False, "fp8_block_gemm": "cutlass"},
     {"NCCL_P2P_DISABLE": "1", "VLLM_BATCH_INVARIANT": "0", "VLLM_USE_DEEP_GEMM": "0"}),
    ("x__bf16__l40s__tp2__b1__i256__o32__mixed__greedy__bi-eager", {}, {"NCCL_P2P_DISABLE": "1", "VLLM_BATCH_INVARIANT": "1"}),
])
def test_tp_row_env_is_the_targets(tmp_path, row_id, target, env):
    out = tmp_path / "w.json"
    a = W.Options(row_id=row_id, hf_repo="org/x", revision="0" * 40, batch=1, max_input=256, max_output=32, case="X",
                  lengths_only=True, check=True, out=str(out), target=json.dumps(target))
    assert W.main(a) == 0
    doc = json.loads(out.read_text())
    assert doc["env_required"] == env and doc["engine_args_required"]["tensor_parallel_size"] == W.row_world(row_id)
