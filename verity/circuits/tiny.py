"""Tiny whole-model configurations, the vLLM tests' `TINY` and two variants of it, at which the registry's bindings and
tests build a served workload's Definitions."""
from __future__ import annotations

TINY = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32, "V": 64, "QKV": 64, "GU": 64,
        "EPS": 1e-5, "ROPE_THETA": 1e4, "MAX_POS": 8, "BIAS": True, "tie_word_embeddings": True, "profile": {}}
TINY_NOBIAS = {**TINY, "BIAS": False, "KVH": 2, "QKV": 96}
#: FP8's per-token quantization reads whole 256-thread CTAs: every reduction length is at least 256
TINY_FP8 = {**TINY, "H": 256, "NL": 1, "NH": 2, "KVH": 1, "D": 128, "I": 256, "V": 16, "QKV": 512, "GU": 512}
