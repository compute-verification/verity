"""What this benchmark takes from the vLLM adapter (`verity_vllm`) that its public interface doesn't offer yet.

The adapter's public interface is the modules ``tests/test_vllm_public_api.py`` lists in ``PUBLIC`` and, from each, the names
in its ``__all__``. Outside code reaches an adapter through that interface only (ARCHITECTURE.md), so in
``benchmarks/inference_only`` this module is the one file that imports past it, each name once, and every other file takes
the name from here (``tests/test_adapter.py``). A name gains a public home when the adapter lists it; then its users import it
from there and its line here goes.

What is left is the served circuits' Boolean gates, whose place the served Definitions' move decides. It imports them when
called, so importing this module stays cheap.

| Here | Non-public home | Used by |
| --- | --- | --- |
| `gate_table` | `verity_vllm.definitions.boolean.gate_table`, after `import verity_vllm.definitions` registers the adapter's registry with the Boolean-versions lookup | built.py |
"""
from __future__ import annotations

from typing import Any

__all__ = ["gate_table"]


def gate_table(fns: Any) -> tuple[dict[str, int], list[str]]:
    """The Boolean gates of each primitive ``fns`` reach, and the primitives the table doesn't price."""
    import verity_vllm.definitions  # noqa: F401  (registers the vLLM registry with the Boolean-versions lookup)
    from verity_vllm.definitions.boolean import gate_table as table

    return table(fns)
