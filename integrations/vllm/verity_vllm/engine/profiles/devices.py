"""The name a row class's GPU reports (`torch.cuda.get_device_name`), for the classes one name covers.

vLLM keys its tuned kernel configs by that name (`get_device_name_as_file_name`: GLM-5.3's fused-MoE tiles, `program.registry.fp8_moe`),
so a GPU-free Build answers its row's (`export_compat.select_cuda_platform`) and reads the config file the class's GPU does.  On a
class not listed the platform answers nothing, and a construction that reads the name fails there by name."""
from __future__ import annotations

import json

from verity_vllm.target_family import device_token_of_row

DEVICE_NAME: dict[str, str] = {"h200": "NVIDIA H200"}


def device_name(workload: str | None) -> str | None:
    """The name a GPU of the workload's row class reports; None for a class `DEVICE_NAME` does not list or without a workload."""
    sweep = (json.load(open(workload)).get("sweep") or {}) if workload else {}
    return DEVICE_NAME.get(device_token_of_row(sweep.get("row_id") or "") or "")
