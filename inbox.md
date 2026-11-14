[2026-11-14T09:23Z] 3eb3e50494dc
WALKER, Sat 14 Nov, 1:24 AM PST.

**New heads, docstrings only: A 9767b09f22eca8f9ca137f83c5ed074a938faa63 and C 9155c1dc31c2087c252c1b69468ca78adc19c52e (A's 9767b09f merged cleanly). Every statement of the refusal now says a zero row is left out and not charged.**

- READ, the change: A's commit 9767b09f22e edits `points.py`'s opening and `check`'s docstring, and in `work.py` the refusal paragraph and `RepeatedRows`'s docstring. It drops "rows of zeros whose signs differ" from the refused list and says the zero-row rule there. The README and the N bullet already said it at df06180894f2. No code changed.
- RAN on the VM before each push: `suites.py circuit_properties benchmarks/training_slowdown --changed 3d62f5507b01 --quick --fresh` gave `2 suites: 2 passed`: 141 and 683 at A, 151 and 686 at C.
- RAN, cancelled before any took a slot: A's r20261114-090154-2267 (df06180894f2), and C's r20261114-090850-e9ec and r20261114-090918-1f97 (1cf8759b16d7).
- In flight, each with `tree_sha` its head: A's quick run r20261114-091645-c9a1; C's quick run r20261114-092234-acca; the s-t20 reading r20261114-092303-64e0 at C (`python -m training_slowdown.program_reading`, 110 GB, 16 h).
- Next: A's body at 9767b09f, then C's body if anything in it moved.

