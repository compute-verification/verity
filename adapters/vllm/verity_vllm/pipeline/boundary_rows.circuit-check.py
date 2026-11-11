"""circuit-check's bindings of `verity_vllm.pipeline.boundary_rows`' Definitions, in `circuit_check.targets`' forms
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


def _stop_roots() -> list[Any]:
    """`ServedStop_v1` at a small class with two stop ids and with none, and at 32 outputs with GLM-5.3's three stop ids
    and a model of 131,072 ids; and `ReplayStopped_v1`: a replay block of two inlined `TokenSelect_v4` with the stop block
    beside them."""
    from verity_vllm.pipeline.boundary_rows import served_stop, stopped

    return [served_stop(4, (3, 7), 64), served_stop(4, (), 64), served_stop(32, (154820, 154827, 154829), 131072),
            stopped(_inlined_roots()[0], served_stop(4, (3, 7), 64))]
