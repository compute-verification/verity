"""Host-side commitment / opening prototype for in-kernel FlashAttention values (R8 lane B-commit, 2026-09-18).

Experimental; imports the C1 commitment engine from `tests/commit/{fasttree,stream_merkle}.py`.  See `encoding.py` (canonical encoding), `layouts.py` (hierarchical and
packed layouts, verification), `derived.py` (derived-value rule), `costs.py` (proof-side cost table), `test_*.py`.
"""
