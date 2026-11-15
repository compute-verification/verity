"""What this benchmark takes from the vLLM adapter (`verity_vllm`) that its public interface doesn't offer yet.

The adapter's public interface is the modules ``tests/test_vllm_public_api.py`` lists in ``PUBLIC`` and, from each, the names
in its ``__all__``. Outside code reaches an adapter through that interface only (ARCHITECTURE.md), so in
``benchmarks/inference_only`` this module is the one file that imports past it, each name once, and every other file takes
the name from here (``tests/test_adapter.py``). A name gains a public home when the adapter lists it; then its users import it
from there and its line here goes.

The modules imported at the top here import neither torch, vLLM nor the PyTorch adapter. A name whose module does is a
function that imports it when called, so importing this module stays cheap.

| Here | Non-public home | Asked of the adapter | Used by |
| --- | --- | --- | --- |
| `Phi` | `verity_vllm.program.predict.phi` | `Phi` (I7: what a serve registers, φ: its model, target and execution) | built.py |
| `registered_of` | `verity_vllm.program.predict.steps` | `Registered`, with `Registered.of` (I7: a serve's registration at φ and a seed) | built.py |
| `PARAM` | `verity_vllm.check.weights_of_record` | `PARAM` (I7: the Program parameter the window circuit's weights are) | built.py, tests |
| `Unexpressible` | `verity_vllm.program.predict.common` | `Unexpressible` (I7: a lowering the composer refuses, with its reason) | built.py, tests |
| `class_circuit` | `verity_vllm.program.predict.window_circuit` | `class_circuit` (I7: a window class's circuit) | built.py, tests |
| `gate_table` | `verity_vllm.definitions.boolean`, after `verity_vllm.definitions` registers the adapter's registry | `gate_table`, its registry registered on import (the served circuits' Boolean gates a primitive) | built.py |
| `dense_generic` | `verity_vllm.engine.profiles.dense_generic` | `checkpoint_entries`, `MANIFEST`, `pinned_hf_config` (I1: the checkpoints the manifests pin, and a pin's config) | policy.py, tests |
| `b1` | `verity_vllm.definitions.registry.b1` | `weights_type` (I7: a model configuration's weights' record type) | policy.py |
"""
from __future__ import annotations

from typing import Any

from verity_vllm.check.weights_of_record import PARAM
from verity_vllm.definitions.registry import b1
from verity_vllm.engine.profiles import dense_generic
from verity_vllm.program.predict.common import Unexpressible
from verity_vllm.program.predict.phi import Phi

__all__ = ["PARAM", "Phi", "Unexpressible", "b1", "class_circuit", "dense_generic", "gate_table", "registered_of"]


def registered_of(phi: Any, seed: int) -> Any:
    """What a serve at ``phi`` registers under ``seed`` (`steps.Registered.of`); imports the PyTorch adapter."""
    from verity_vllm.program.predict.steps import Registered

    return Registered.of(phi, seed)


def class_circuit(registered: Any, cls: Any) -> Any:
    """The circuit of the window class ``cls`` over ``registered``; raises `Unexpressible` with the composer's reason.
    Imports the PyTorch adapter."""
    from verity_vllm.program.predict.window_circuit import class_circuit as lower

    return lower(registered, cls)


def gate_table(fns: Any) -> tuple[dict[str, int], list[str]]:
    """The Boolean gates of each primitive ``fns`` reach, and the primitives the table doesn't price."""
    import verity_vllm.definitions  # noqa: F401  (registers the vLLM registry with the Boolean-versions lookup)
    from verity_vllm.definitions.boolean import gate_table as table

    return table(fns)
