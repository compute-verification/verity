"""circuit-check's bindings of `verity.commitments.gates.embedding`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _embedding_roots() -> list[Any]:
    """The embedding's row read (`verity.commitments.gates.embedding`) at its smallest binding, two one-word rows: a 16-bit
    row's digest, its hm96 leaf, a one-level frame-v3 read over ``hm96-sha512/row-seg/v1`` leaves and the selection; and the
    selection at the deployed bindings (Llama-3.2-1B's and Llama-3.1-8B's vocabulary and hidden sizes), whose token comparison
    and fill a small V or H doesn't reach. The deployed read repeats these parts at their widths (a full row-seg segment, the
    read's 17 levels), each checked at its smallest binding here and in its own module's bindings, and its gates are pinned
    in the tests (``READS``).  The read of a history table (`verity_vllm.definitions.registry.dsa_topk_reads`: a capacity of
    2^17 rows of GLM-5.3's 576-word kv row) is checked whole at its served binding, held to its digest pin, with its
    selection. The opening (`opening`, ``EmbeddingRead_v3``) at its smallest binding, with its guard at the deployed
    vocabularies, whose comparison a small V doesn't reach, and its one-word selection."""
    from verity.circuits.defs import bind
    from verity.commitments.gates.embedding import EmbeddingReadGuard, EmbeddingReadSelect, EmbeddingReadWord, opening, read

    return [read(2, 1), bind(EmbeddingReadSelect, V=128256, H=2048), bind(EmbeddingReadSelect, V=128256, H=4096),
            read(1 << 17, 576), bind(EmbeddingReadSelect, V=1 << 17, H=576), opening(2, 1),
            bind(EmbeddingReadGuard, V=128256), bind(EmbeddingReadGuard, V=1 << 17), bind(EmbeddingReadWord)]
