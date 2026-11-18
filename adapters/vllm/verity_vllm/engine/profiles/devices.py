"""The name a workload class's GPU reports (`torch.cuda.get_device_name`), for the classes one name covers, and the tuned kernel configs vLLM
keys by it.

vLLM keys its tuned kernel configs by that name (`get_device_name_as_file_name`: GLM-5.3's fused-MoE tiles, `program.registry.fp8_moe`),
so a GPU-free Build answers its target's (`export_compat.select_cuda_platform`: a traced request Build's `--device-name`, else its
row's) and reads the config file that GPU does.  On a class not listed the platform answers nothing, and a construction that reads
the name fails there by name.  `FUSED_MOE_CONFIGS` holds
the files a prediction reads without vLLM (`predict`).  Both names come from the GPU card records (`config.CARD_FILES`): a class
is the record whose `short` is its token and reports that record's `driver` name, and a reported name is registered on the record
whose `driver` holds it (`card_sku`, through `verity.device`'s reader)."""
from __future__ import annotations

import json
from functools import cache

from verity.device import gpu_card_reporting, load_card, skus
from verity_vllm.config import CARD_FILES, workload_file
from verity_vllm.target_family import device_token_of_row

__all__ = ["card_sku", "class_device_name", "device_name", "device_name_of_row", "fused_moe_config"]


@cache
def _gpu_records() -> tuple[dict, ...]:
    return tuple(load_card(sku, directory=CARD_FILES) for sku in skus("gpu", directory=CARD_FILES))


def _driver_names(record: dict) -> list[str]:
    return [d["value"] for d in record.get("driver", ())]


def card_sku(name: str) -> str:
    """The SKU of the one GPU card file whose `driver` holds `name`, the name the device reports (`verity.device.gpu_card_reporting`);
    refused by name (KeyError) where none or several do."""
    return gpu_card_reporting(name, directory=CARD_FILES)["sku"]


def class_device_name(token: str) -> str | None:
    """The name a GPU of row class `token` reports: the one `driver` name of the one GPU card record whose `short` is `token`; None
    for a class no record names, or whose record states no one name."""
    found = [r for r in _gpu_records() if r.get("short") == token]
    names = _driver_names(found[0]) if len(found) == 1 else []
    return names[0] if len(names) == 1 else None

#: vLLM d9105ea80's tuned fused-MoE configs (model_executor/layers/fused_moe/configs/<file>, `get_config_file_name`), batch size ->
#: (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_warps, num_stages); the file below is sha256 ade035ab90cbce92...
FUSED_MOE_CONFIGS: dict[str, dict[int, tuple[int, int, int, int, int, int]]] = {
    "E=256,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json": {
        1: (8, 32, 128, 1, 4, 4), 2: (8, 32, 128, 1, 4, 5), 4: (8, 64, 128, 1, 4, 4), 8: (16, 64, 128, 32, 4, 3),
        16: (16, 128, 128, 16, 4, 3), 24: (16, 128, 128, 16, 4, 4), 32: (16, 128, 128, 16, 4, 5), 48: (16, 128, 128, 64, 4, 3),
        64: (16, 128, 128, 64, 4, 3), 96: (16, 128, 128, 16, 4, 3), 128: (16, 128, 128, 32, 4, 3), 256: (16, 128, 128, 32, 4, 3),
        512: (16, 128, 128, 16, 4, 3), 1024: (64, 128, 128, 32, 4, 4), 1536: (64, 128, 128, 32, 4, 4), 2048: (64, 128, 128, 32, 4, 4),
        3072: (64, 128, 128, 32, 4, 4), 4096: (64, 128, 128, 16, 4, 3),
    },
}


def device_name_of_row(row_id: str) -> str | None:
    """The name a GPU of the workload's class reports (`class_device_name`); None for a class no card record names."""
    token = device_token_of_row(row_id or "")
    return class_device_name(token) if token else None


def device_name(workload: str | None) -> str | None:
    """The name a GPU of the workload's row class reports; None for a class no card record names, for a workload with no
    sweep plan, and without a workload.  A stored workload is made present and checked first (`config.workload_file`)."""
    held = json.loads(workload_file(workload).read_text()).get("sweep") if workload else None
    sweep = held if held else {}
    return device_name_of_row(sweep.get("row_id") or "")


def fused_moe_config(E: int, N: int, device: str, dtype: str, block: tuple[int, int], M: int) -> dict[str, int] | None:
    """The tile vLLM's `try_get_optimal_moe_config` takes for M tokens from the tuned file of (E experts, N = a rank's expert width,
    the device's name, dtype, block shape): the entry at the batch size nearest M, the smaller on a tie (fused_moe.py:1442-1447);
    None when `FUSED_MOE_CONFIGS` has no such file."""
    name = f"E={E},N={N},device_name={device.replace(' ', '_')},dtype={dtype},block_shape=[{block[0]},{block[1]}].json"
    table = FUSED_MOE_CONFIGS.get(name)
    if table is None:
        return None
    key = min(table, key=lambda m: (abs(m - M), m))
    return dict(zip(("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "num_warps", "num_stages"), table[key]))
