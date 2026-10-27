"""circuit-check's bindings of `verity.ml.definitions.registry.fa3_split`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import fa3_split, hopper

    return [
        # GLM-5.3's split dense-MHA prefill (DsaDenseMha[vllm-dsa] where FA3 splits) at its tile's 80-key blocks, one head: row 160
        # of a prefill at one block a split, three splits of which the first is below the tile's MASKED_FROM and the last partial.
        # Head 32 keeps the catalog's cost down; the served head 256 is the same id at D=256, which `circuit-check` takes on demand
        ("AttentionSplit_v1", lambda: bind(fa3_split.AttentionSplit, T=161, NH=1, KVH=1, D=32, BN=80, MASKED_FROM=1, PER=1,
                                           DOT=hopper.HopperBF16WgmmaDot16, INV=hopper.Fa3InvSum)),
    ]
