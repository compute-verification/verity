"""circuit-check's bindings of `verity_vllm.pipeline.boundary_rows`' Definition, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _inlined_roots() -> list[Any]:
    """`TemplateInlined_v1` of `TokenSelect_v4` at four chunks, the last one short, at N = 2: an inlined check template
    whose body calls three templates (its two chunk widths and its combine)."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import dead_free as DF
    from verity_vllm.pipeline.boundary_rows import inlined

    return [inlined(bind(DF.TokenSelectV4, V=200), 2)]
