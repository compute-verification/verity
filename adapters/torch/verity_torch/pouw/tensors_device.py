"""`tensors.Tensors` over PyTorch's tensors (`TENSORS`): what the PyTorch adapter hands PoUW's serving code (`tensors.use`)."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

#: `tensors`' element type names
DTYPES = {"uint8": torch.uint8, "int8": torch.int8, "int16": torch.int16, "uint16": torch.uint16, "int32": torch.int32,
          "int64": torch.int64, "float32": torch.float32, "bfloat16": torch.bfloat16}


def _device(like: Any) -> torch.device | None:
    if like is None:
        return None
    return like.device if isinstance(like, torch.Tensor) else torch.device(like)


class TorchTensors:
    """`tensors.Tensors` on torch tensors: a buffer is a tensor, a device a tensor's or a `torch.device`'s spelling."""

    def is_buffer(self, x: Any, /) -> bool:
        return isinstance(x, torch.Tensor)

    def host(self, t: Any, /) -> np.ndarray:
        t = t.detach()
        return t.view(torch.uint8).cpu().numpy().view("<u2") if t.dtype == torch.uint16 else t.cpu().numpy()

    def put(self, a: np.ndarray, like: Any = None, /, *, as_like: bool = False) -> Any:
        t = torch.from_numpy(a)
        dev, dtype = _device(like), like.dtype if as_like else None
        if dev is None or dev.type == "cpu":
            return t if dtype is None else t.to(dtype)
        return t.pin_memory().to(device=dev, dtype=dtype, non_blocking=True)

    def zeros(self, shape: tuple[int, ...], dtype: str, like: Any = None, /) -> Any:
        if like is None:
            if torch.cuda.is_available():
                return torch.zeros(shape, dtype=DTYPES[dtype], pin_memory=True).numpy()
            return np.zeros(shape, dtype)
        return torch.zeros(shape, dtype=DTYPES[dtype], device=_device(like))

    def view(self, t: Any, dtype: str, /) -> Any:
        return t.view(DTYPES[dtype])

    def pointer(self, t: Any, /) -> int:
        return int(t.data_ptr())

    def stream(self, like: Any = None, /) -> int:
        return int(torch.cuda.current_stream(_device(like)).cuda_stream)

    def capability(self, like: Any = None, /) -> tuple[int, int]:
        major, minor = torch.cuda.get_device_capability(_device(like))
        return int(major), int(minor)

    def copy_to_host(self, t: Any, out: np.ndarray, /) -> Callable[[], None]:
        stream = torch.cuda.current_stream(t.device)
        torch.from_numpy(out).copy_(t, non_blocking=True)
        # the copy is still queued when this returns: the allocator gives its source to no other stream till then
        t.record_stream(stream)
        copied = torch.cuda.Event(blocking=True)
        copied.record(stream)
        return copied.synchronize


TENSORS = TorchTensors()

__all__ = ["DTYPES", "TENSORS", "TorchTensors"]
