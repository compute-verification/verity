"""Host-side commitment / opening prototype for in-kernel FlashAttention values (R8 lane B-commit, 2026-09-18).

Experimental; imports the C0/C1 commitment engine from `verity_vllm/commit/{fasttree,semantic_layout,stream_merkle}.py` and the production opening
verifier core from `verity_vllm/commit/merkle.py`.  See `encoding.py` (canonical encoding), `layouts.py` (hierarchical and
packed layouts, verification), `derived.py` (derived-value rule), `costs.py` (proof-side cost table), `test_*.py`.
"""
