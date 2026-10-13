"""The sweep workload generator: a TP row's `env_required` names NCCL_P2P_DISABLE on a SYS target only."""
from __future__ import annotations

import json

import pytest

from verity_vllm.config import WORKLOADS, row_target, row_world
from verity_vllm.engine.profiles.transport import P2P_SYS_TARGETS, row_p2p_env
from verity_vllm.pipeline import workload as W


@pytest.mark.parametrize("row_id, p2p", [
    ("x__bf16__l40s__tp2__b1__i256__o32__mixed__greedy__bi-eager", {"NCCL_P2P_DISABLE": "1"}),
    ("x__bf16__rtxpro6000__tp8__b1__i256__o32__mixed__greedy__bi-eager", {"NCCL_P2P_DISABLE": "1"}),
    ("x__bf16__h200__tp8__b1__i256__o32__mixed__greedy__bi-eager", {}),
    ("x__bf16__h100__tp2__b1__i256__o32__mixed__greedy__bi-eager", {}),
])
def test_a_tp_workload_requires_nccl_p2p_disable_on_a_sys_target_only(tmp_path, row_id, p2p):
    out = tmp_path / "w.json"
    a = W.Options(row_id=row_id, hf_repo="org/x", revision="0" * 40, batch=1, max_input=256, max_output=32, case="X",
                  lengths_only=True, check=True, out=str(out))
    assert W.main(a) == 0
    assert json.loads(out.read_text())["env_required"] == {**p2p, "VLLM_BATCH_INVARIANT": "1"}


def test_every_tp_workload_disables_p2p_exactly_on_a_sys_target():
    """What each TP row's stages set (`row_tp.TpRow.p2p_env`: `env_required`, else the generator's rule for a workload written
    before it had one): NCCL_P2P_DISABLE=1 on every L40S and RTX PRO 6000 row, and on no H box's."""
    rows = sorted(p for p in WORKLOADS.glob("*.json") if row_world(p.stem) > 1)
    assert rows
    for p in rows:
        req = json.loads(p.read_text()).get("env_required")
        p2p = (row_p2p_env(p.stem) if req is None else req).get("NCCL_P2P_DISABLE")
        assert (p2p == "1") == (row_target(p.stem) in P2P_SYS_TARGETS) and p2p in ("1", None), p.name
    assert "glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager" in {p.stem for p in rows}
