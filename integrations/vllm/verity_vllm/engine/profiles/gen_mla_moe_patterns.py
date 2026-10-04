"""The `mla-moe` κ patterns (GLM-4.7-Flash, `Glm4MoeLiteForCausalLM` on vLLM d9105ea80, TRITON_MLA decode / FA2 prefill)."""

from __future__ import annotations

from typing import Any


def with_mla_moe_patterns(patterns: list, C: dict[str, Any]) -> list:
    return list(patterns)
