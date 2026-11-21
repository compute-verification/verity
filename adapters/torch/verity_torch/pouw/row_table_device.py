"""One CUDA device's copy of a row-table kernel library with its side streams (`DeviceLibrary`), over the host build of
`verity.protocols.accounting.compute.pouw.prover.serving.row_table_kernels`: H1's device tables, registered rows and receipts
build on it, and so does the vLLM adapter's Commit (its ``commit.row_tables_device``)."""
from __future__ import annotations

import ctypes
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from verity.device import compute_capability, gpu_card_reporting
from verity.protocols.accounting.compute.pouw.prover.serving.row_table_kernels import device_lib, hm_tables, leaf_prefixes

__all__ = ["CARD_FILES", "DeviceLibrary", "arch", "side_stream"]

#: the GPU card files (`verity.device`), whose ``arch`` is the target every library here is built for
CARD_FILES = Path(__file__).resolve().parents[4] / "calibration/hardware/cards"


def arch(name: str) -> str:
    """The target a library is built for on the device reporting ``name`` (torch's ``get_device_name``): its card file's
    ``arch`` (`verity.device.gpu_card_reporting`) without ``sm_``, such as ``90a``; a device no card file names is refused."""
    card = gpu_card_reporting(name, directory=CARD_FILES)
    compute_capability(card)
    return card["arch"].removeprefix("sm_")


class DeviceLibrary:
    """One CUDA device's build of a kernel library over hm96-sha512 (``lib(arch)``: `row_table_kernels.device_lib`, or H1's
    `tables_device` and `registered_device`'s), its copies of the byte tables and midstates (the library's ``midstates``
    kernel over `leaf_prefixes`), the side streams its kernels hash on (``stream`` the first, `next_stream` each in turn) and the
    identity rank order its tables share (`ranks`).  `of` builds one per library and device, once a process."""

    _of: dict[tuple[Callable[[str], ctypes.CDLL], int], DeviceLibrary] = {}

    def __init__(self, torch: Any, device: Any, lib: Callable[[str], ctypes.CDLL], midstates: str, streams: int) -> None:
        self.lib = lib(arch(torch.cuda.get_device_name(device)))
        self.streams, self.turn = [torch.cuda.Stream(device) for _ in range(streams)], 0
        self.stream = self.streams[0]
        with torch.cuda.stream(self.stream):
            self.hm = torch.from_numpy(hm_tables().view(np.int64)).to(device)
            pre = torch.from_numpy(leaf_prefixes().view(np.int64)).to(device)
            self.ms = torch.empty(16, dtype=torch.int64, device=device)
            rc = getattr(self.lib, midstates)(self.stream.cuda_stream, pre.data_ptr(), self.ms.data_ptr())
        if rc:
            raise RuntimeError(f"{midstates}: CUDA error {rc}")
        self.stream.synchronize()
        self.perm = torch.arange(0, dtype=torch.int32, device=device)

    @classmethod
    def of(cls, torch: Any, index: int, lib: Callable[[str], ctypes.CDLL] = device_lib, midstates: str = "rt_midstates",
           streams: int = 1) -> DeviceLibrary:
        if (lib, index) not in cls._of:
            cls._of[lib, index] = cls(torch, torch.device("cuda", index), lib, midstates, streams)
        return cls._of[lib, index]

    def next_stream(self) -> Any:
        self.turn = (self.turn + 1) % len(self.streams)
        return self.streams[self.turn]

    def ranks(self, torch: Any, n: int) -> Any:
        """The identity rank order over at least ``n`` rows."""
        if len(self.perm) < n:
            self.perm = torch.arange(max(n, 2 * len(self.perm)), dtype=torch.int32, device=self.perm.device)
        return self.perm


def side_stream(torch: Any, device: Any = None) -> Any:
    """The side stream steps on ``device`` (default: the current CUDA device) hash on, ``torch`` the workload's library."""
    return DeviceLibrary.of(torch, torch.cuda.current_device() if device is None else torch.device(device).index).stream
