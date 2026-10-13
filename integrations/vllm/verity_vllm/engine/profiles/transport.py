"""NCCL's transport and vLLM's custom all-reduce per device class: what a tensor-parallel row's engines run with.

On a box whose GPUs reach each other only across PCIe and the CPU interconnect (`nvidia-smi topo -m`: SYS), NCCL's P2P transport
hangs vLLM's all-reduce (#70 / #75), so a TP workload for such a target requires NCCL_P2P_DISABLE=1 in its `env_required` and
``disable_custom_all_reduce: true`` in its `engine_args_required`, which `pipeline.workload` writes and `pipeline.row_tp.TpRow`
applies.  Every other target runs vLLM's own transport and its custom all-reduce (Daniel, 13 Oct: standard vLLM on H boxes).

A workload written before the row followed it falls back differently for the two:

* no `env_required` (four L40S workloads predate it): the generator's value for the row's target, `row_p2p_env`;
* no ``engine_args_required.disable_custom_all_reduce``: True, what every TP row's engines ran before (top, 13 Oct),
  `workload_custom_all_reduce_off`.  The generator always writes the field, so only an old workload omits it.
"""
from __future__ import annotations

from typing import Any

from verity_vllm.config import row_target

#: Row-id target tokens whose TP boxes are SYS.
P2P_SYS_TARGETS: frozenset[str] = frozenset({"l40s", "rtxpro6000"})


def row_p2p_env(row: str) -> dict[str, str]:
    """A TP row's ``env_required`` NCCL_P2P_DISABLE by its target token: ``{"NCCL_P2P_DISABLE": "1"}`` on a SYS target, else none."""
    return {"NCCL_P2P_DISABLE": "1"} if row_target(row) in P2P_SYS_TARGETS else {}


def row_custom_all_reduce_off(row: str) -> bool:
    """A TP row's ``engine_args_required.disable_custom_all_reduce`` by its target token: True on a SYS target, else False (vLLM's
    default)."""
    return row_target(row) in P2P_SYS_TARGETS


def workload_custom_all_reduce_off(workload: dict[str, Any]) -> bool:
    """The ``disable_custom_all_reduce`` a TP row's engines run with: its workload's ``engine_args_required`` value, else True."""
    return bool((workload.get("engine_args_required") or {}).get("disable_custom_all_reduce", True))
