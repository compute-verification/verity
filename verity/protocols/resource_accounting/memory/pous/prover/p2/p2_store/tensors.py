"""The tensors of PoUS's weight store (`verity/protocols/resource_accounting/memory/pous/prover/weight_store.py`, which imports no torch):
a tensor's bytes as a flat uint8 view (`flat`), ``C*``'s rows on the weights' device (`rows`, `put`), the device's index
for the GPU encoder (`device_index`) and a decode's bytes into its buffer (`fill`). No torch until one is called."""
from __future__ import annotations

from typing import Any


def flat(t: Any) -> Any:
    """``t``'s bytes as a flat uint8 view."""
    import torch
    return t.detach().reshape(-1).view(torch.uint8)


def device_index(device: Any) -> int:
    """The CUDA device's index, the current one where ``device`` names none."""
    import torch
    return torch.cuda.current_device() if device.index is None else device.index


def rows(blocks: int, block_bytes: int, device: Any) -> Any:
    """``C*``'s rows, ``[blocks, block_bytes]`` uint8 on ``device``, unfilled."""
    import torch
    return torch.empty(blocks, block_bytes, dtype=torch.uint8, device=device)


def put(rows: Any, first: int, data: bytes, block_bytes: int) -> None:
    """``data``, whole blocks of ``C*``, into ``rows`` from row ``first``."""
    import torch
    n = len(data) // block_bytes
    rows[first:first + n].copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8).view(n, block_bytes))


def fill(out: Any, data: bytes) -> None:
    """``data`` into ``out``, a flat uint8 buffer of its length."""
    import torch
    out.copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))
