"""E1 (EOD 2026-09-11): interior value production for one declared TINY request with the `verity.ir` reference interpreter.

`producer.py` streams every computed gate value of one RU (value record JSONL, brief interface 2), `compare_boundaries.py`
hashes every operator-boundary row the interpreter produces exactly as the store does and compares it with the committed leaf,
`rate.py` reports the measured gates/s, wall and peak RSS.  Layout (`layout.py`, `tree.py`) is E2's; the position oracle is E3's.
"""
