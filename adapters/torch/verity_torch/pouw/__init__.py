"""PoUW's serving code over PyTorch: what `verity.protocols.resource_accounting.compute.pouw.prover.serving` leaves to the workload.

Importing this package hands its implementations in (`serving.tensors.use`, `use_h1`): `tensors_device.TENSORS`, the
tensors on the device (pinned host copies, streams, raw pointers), and `h1.h1_device.H1`, H1's call tables, registered rows,
trace and records on the device. Each is imported when first asked for, so this import loads no torch.

The rest is the workload's device code, which `serving` names nowhere:

- `pearl_c`: Pearl-C's run on the device (`PearlCDeviceRun`, `pearl_c_weight_roots`), its mixins (`lanes_device`, `ends`,
  `graphs`, `passes`, `grouped`, `whole`), host staging (`host`), the served y's trees (`served_y_device`) and the Triton
  screen (`screen`), and the engine's per-step hook (`hook_steps`);
- `h1`: the receipts (`Receipts`), and the device halves of the registered rows, the call tables (`h1_tables_available`)
  and the records;
- `lead.rows_device`: the lead's words and A_j hashed on the device;
- `row_table_device`: a CUDA device's copy of the row-table kernels with its side streams (`RowTableLibrary`,
  `row_table_side_stream`), which H1's device halves and the vLLM adapter's Commit build on;
- `credit_gate_device`: the credit gate's trace of the dispatcher's ops.
"""
from __future__ import annotations

import importlib
from typing import Any

from verity.protocols.resource_accounting.compute.pouw.prover.serving import tensors as _tensors

__all__ = ["PearlCDeviceRun", "Receipts", "RowTableLibrary", "h1_tables_available", "hook_steps", "pearl_c_weight_roots",
           "row_table_side_stream"]

_LAZY = {
    "PearlCDeviceRun": ("verity_torch.pouw.pearl_c.run_device", "PearlCDeviceRun"),
    "pearl_c_weight_roots": ("verity_torch.pouw.pearl_c.run_device", "weight_roots"),
    "hook_steps": ("verity_torch.pouw.pearl_c.passes", "hook_steps"),
    "Receipts": ("verity_torch.pouw.h1.receipts", "Receipts"),
    "h1_tables_available": ("verity_torch.pouw.h1.tables_device", "available"),
    "RowTableLibrary": ("verity_torch.pouw.row_table_device", "DeviceLibrary"),
    "row_table_side_stream": ("verity_torch.pouw.row_table_device", "side_stream"),
}


def _tensors_impl() -> Any:
    from verity_torch.pouw.tensors_device import TENSORS
    return TENSORS


def _h1_impl() -> Any:
    from verity_torch.pouw.h1.h1_device import H1
    return H1


_tensors.use(_tensors_impl)
_tensors.use_h1(_h1_impl)


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attr = _LAZY[name]
    return getattr(importlib.import_module(module), attr)
