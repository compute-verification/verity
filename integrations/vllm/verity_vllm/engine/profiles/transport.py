"""NCCL's transport per device class: the row-id targets whose tensor-parallel rows run with NCCL_P2P_DISABLE=1.

On a box whose GPUs reach each other only across PCIe and the CPU interconnect (`nvidia-smi topo -m`: SYS), NCCL's P2P transport
hangs vLLM's all-reduce (#70 / #75), so a TP workload for such a target requires NCCL_P2P_DISABLE=1 in its `env_required`, which
`pipeline.workload` writes and `pipeline.row_tp.TpRow` applies.  Every other target runs vLLM's own transport (Daniel, 13 Oct:
standard vLLM on H boxes).
"""
from __future__ import annotations

from verity_vllm.config import row_target

#: Row-id target tokens whose TP boxes are SYS.
P2P_SYS_TARGETS: frozenset[str] = frozenset({"l40s", "rtxpro6000"})


def row_p2p_env(row: str) -> dict[str, str]:
    """A TP row's ``env_required`` NCCL_P2P_DISABLE by its target token: ``{"NCCL_P2P_DISABLE": "1"}`` on a SYS target, else none."""
    return {"NCCL_P2P_DISABLE": "1"} if row_target(row) in P2P_SYS_TARGETS else {}
