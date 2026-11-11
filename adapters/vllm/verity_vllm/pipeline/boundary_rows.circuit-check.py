"""circuit-check's bindings of `verity_vllm.pipeline.boundary_rows`' Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _inlined_roots() -> list[Any]:
    """`TemplateInlined_v1` of `TokenSelect_v4` at four chunks, the last one short, at N = 2: an inlined check template
    whose body calls three templates (its two chunk widths and its combine)."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import dead_free as DF
    from verity_vllm.pipeline.boundary_rows import inlined

    return [inlined(bind(DF.TokenSelectV4, V=200), 2)]


def _stop_roots() -> list[Any]:
    """`ServedStop_v1` at a small class with two stop ids and with none, and `ReplayStopped_v1`: a replay block of two
    inlined `TokenSelect_v4` with the stop block beside them."""
    from verity_vllm.pipeline.boundary_rows import served_stop, stopped

    return [served_stop(4, (3, 7), 64), served_stop(4, (), 64), stopped(_inlined_roots()[0], served_stop(4, (3, 7), 64))]


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`ServedStop_v1` at GLM-5.3's three stop ids and a model of 131,072 ids: at 32 outputs, and at a class of 1,024."""
    from verity_vllm.pipeline.boundary_rows import served_stop

    glm = (154820, 154827, 154829)
    return [("ServedStop_v1:32:3", lambda: served_stop(32, glm, 131072)),
            ("ServedStop_v1:1024:3", lambda: served_stop(1024, glm, 131072))]
