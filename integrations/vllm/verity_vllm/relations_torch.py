"""``same_runtime`` checkers that need torch.  Imported only on demand (:func:`install`).

These re-run the *registered runtime operator* on authenticated inputs and return its bytes.
They establish that operands and result agree under that operator; they are not an independent
numerical model (contract section 7), and every one of them sets ``includes_full_op=True``.

W1 extends this module on the GPU pod with the vLLM custom ops (``rms_norm``, ``rotary_embedding``,
the attention backend, the quantized GEMMs); the two functions here show the pattern and run on
CPU so the harness can test the wiring without a GPU.
"""

from __future__ import annotations

from typing import Any

from . import dtypes
from .description import NodeInstance
from .relations import SAME_RUNTIME, Inputs, RelationError, need, register

_TORCH_DTYPES = {
    "bfloat16": "bfloat16",
    "float16": "float16",
    "float32": "float32",
    "int32": "int32",
    "int64": "int64",
}


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "verity_vllm.relations_torch needs torch; install it on the pod (the pure-numpy "
            "`independent` checkers in verity_vllm.relations do not)"
        ) from error
    return torch


def to_tensor(value: Any, device: str = "cpu") -> Any:
    torch = _torch()
    if value.dtype not in _TORCH_DTYPES:
        raise RelationError(f"{value.canonical_id}: no torch mapping for {value.dtype}")
    view = dtypes.bytes_to_array(value.dtype, value.shape, value.data)
    tensor = torch.from_numpy(view)
    if value.dtype == "bfloat16":
        tensor = tensor.view(torch.bfloat16)
    return tensor.to(device)


def tensor_bytes(tensor: Any) -> bytes:
    torch = _torch()
    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def install(*, device: str = "cpu", replace: bool = False) -> list[str]:
    """Register the torch-backed ``same_runtime`` checkers; returns the ids registered."""
    torch = _torch()
    registered: list[str] = []

    @register(
        "residual_add.same_runtime.v1",
        klass=SAME_RUNTIME,
        op_type="residual_add",
        includes_full_op=True,
        doc=f"torch.add on bf16 tensors ({device}); torch {torch.__version__}",
        owner="W2",
        replace=replace,
    )
    def residual_add(inst: NodeInstance, inputs: Inputs) -> bytes:
        a, b = need(inputs, "a"), need(inputs, "b")
        return tensor_bytes(torch.add(to_tensor(a, device), to_tensor(b, device)))

    registered.append("residual_add.same_runtime.v1")

    @register(
        "silu_mul.same_runtime.v1",
        klass=SAME_RUNTIME,
        op_type="silu_mul",
        includes_full_op=True,
        doc=f"F.silu(gate) * up on the concatenated [2I] row ({device}); torch {torch.__version__}",
        owner="W2",
        replace=True,
    )
    def silu_mul(inst: NodeInstance, inputs: Inputs) -> bytes:
        x = to_tensor(need(inputs, "gate_up"), device)
        half = x.shape[-1] // 2
        gate, up = x[..., :half], x[..., half:]
        return tensor_bytes(torch.nn.functional.silu(gate) * up)

    registered.append("silu_mul.same_runtime.v1")
    return registered
