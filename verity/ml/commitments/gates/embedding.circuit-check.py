"""circuit-check's bindings of `verity.ml.commitments.gates.embedding`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _embedding_roots() -> list[Any]:
    """The embedding's row read (`verity.ml.commitments.gates.embedding`) at its smallest binding, two one-word rows: a 16-bit
    row's digest, its hm96 leaf, a one-level frame-v3 read over ``hm96-sha512/row-seg/v1`` leaves and the selection; and the
    selection at the deployed bindings (Llama-3.2-1B's and Llama-3.1-8B's vocabulary and hidden sizes), whose token comparison
    and fill a small V or H doesn't reach. The deployed read repeats these parts at their widths (a full row-seg segment, the
    read's 17 levels), each checked at its smallest binding here and in its own module's bindings, and its gates are pinned
    in the tests (``READS``)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.commitments.gates.embedding import EmbeddingReadSelect, read

    return [read(2, 1), bind(EmbeddingReadSelect, V=128256, H=2048), bind(EmbeddingReadSelect, V=128256, H=4096)]
