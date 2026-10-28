"""NCCL's transport and vLLM's custom all-reduce per device class: what a tensor-parallel workload's engines run with.

On a box whose GPUs reach each other only across PCIe and the CPU interconnect (`nvidia-smi topo -m`: SYS), NCCL's P2P transport
hangs vLLM's all-reduce, so a TP workload for such a target requires NCCL_P2P_DISABLE=1 in its `engine.env` and
``disable_custom_all_reduce: true`` in its `engine.args`, which `pipeline.workload` writes and `pipeline.row_tp.TpRow` applies.  Every
other target, the H boxes among them, runs vLLM's own transport and its custom all-reduce, as standard vLLM does.  A TP workload's file states
both (`config.RowConfig` refuses one without ``disable_custom_all_reduce``); nothing falls back to a default.
"""
from __future__ import annotations

from verity_vllm.config import row_target

#: Row-id target tokens whose TP boxes are SYS.
P2P_SYS_TARGETS: frozenset[str] = frozenset({"l40s", "rtxpro6000"})
#: Why a TP workload runs with what `row_p2p_env` and `row_custom_all_reduce_off` give it (a served Commit's record: `engine.served`).
REASON = ("NCCL's P2P transport hangs vLLM's all-reduce on a box whose GPUs reach each other only across PCIe and the CPU "
          "interconnect (SYS), so its TP rows run with NCCL_P2P_DISABLE=1 and the custom all-reduce off (#70 / #75); every other "
          "box runs vLLM's own transport")


def row_p2p_env(row: str) -> dict[str, str]:
    """A TP workload's ``engine.env`` NCCL_P2P_DISABLE by its target token: ``{"NCCL_P2P_DISABLE": "1"}`` on a SYS target, else none."""
    return {"NCCL_P2P_DISABLE": "1"} if row_target(row) in P2P_SYS_TARGETS else {}


def row_custom_all_reduce_off(row: str) -> bool:
    """A TP workload's ``engine.args.disable_custom_all_reduce`` by its target token: True on a SYS target, else False (vLLM's
    default)."""
    return row_target(row) in P2P_SYS_TARGETS
