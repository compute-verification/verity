"""`verity.circuits.codec`'s specialization encoding and partition values, which a row's Program graph
(`verity_vllm.pipeline.program_graph`) reads."""
from __future__ import annotations

from verity.circuits.codec import WORD_EXACT, WORD_SAMPLES, encode_specialization, word_values

__all__ = ["WORD_EXACT", "WORD_SAMPLES", "encode_specialization", "word_values"]
