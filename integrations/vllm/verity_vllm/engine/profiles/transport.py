"""NCCL's transport and vLLM's custom all-reduce per device class: what a tensor-parallel row's engines run with.

On a box whose GPUs reach each other only across PCIe and the CPU interconnect (`nvidia-smi topo -m`: SYS), NCCL's P2P transport
hangs vLLM's all-reduce (#70 / #75), so a TP row for such a target requires NCCL_P2P_DISABLE=1 in its `engine.env` and
``disable_custom_all_reduce: true`` in its `engine.args`, which `pipeline.workload` writes and `pipeline.row_tp.TpRow` applies.  Every
other target runs vLLM's own transport and its custom all-reduce (Daniel, 13 Oct: standard vLLM on H boxes).  A TP row's file states
both (`config.RowConfig` refuses one without ``disable_custom_all_reduce``); nothing falls back to a default.
"""
from __future__ import annotations

from verity_vllm.config import row_target

#: Row-id target tokens whose TP boxes are SYS.
P2P_SYS_TARGETS: frozenset[str] = frozenset({"l40s", "rtxpro6000"})


def row_p2p_env(row: str) -> dict[str, str]:
    """A TP row's ``engine.env`` NCCL_P2P_DISABLE by its target token: ``{"NCCL_P2P_DISABLE": "1"}`` on a SYS target, else none."""
    return {"NCCL_P2P_DISABLE": "1"} if row_target(row) in P2P_SYS_TARGETS else {}


def row_custom_all_reduce_off(row: str) -> bool:
    """A TP row's ``engine.args.disable_custom_all_reduce`` by its target token: True on a SYS target, else False (vLLM's
    default)."""
    return row_target(row) in P2P_SYS_TARGETS
