[2026-11-14T09:09Z] d4d9a273597e
WALKER, Sat 14 Nov, 1:10 AM PST.

**C's head is pushed: 1cf8759b16d72ff62d4974db03600a884f7092db (cursor/trn145-mask-structure-eb6e), C's 7e94d534 with A's df06180894f2 merged cleanly. Its quick run and the one s-t20 reading are launched.**

- RAN on the VM at this head: `suites.py circuit_properties benchmarks/training_slowdown --changed 3d62f5507b01 --quick --fresh` gave `2 suites: 2 passed` (circuit-properties 151, training_slowdown 686, with `test_value_reading.py` and `test_program_reading.py` run whole).
- RAN, cancelled before either took a slot: r20261114-083506-23be and r20261114-083504-83ab, both at 7e94d534.
- In flight, `research run --on auto`, `tree_sha` 1cf8759b16d7: r20261114-090850-e9ec, `suites.py --changed 3d62f5507b01 --quick`; r20261114-090918-1f97, `python -m training_slowdown.program_reading --out $OUT/reading.json` (s-t20's shape, 110 GB, 16 h), waiting for a slot. A's quick run, r20261114-090154-2267 at df06180894f2, is waiting too.
- READ, on the charge's share (for READCIRC's question): a site's share comes only from its own claimed operands' rows at its own node, `Tally.zero` keyed by (summary, node, argument). It is the most of any activation the pass evaluates for that summary, applied to all of the site's counts, so N only falls. If "no site charged more than its own zero rows allow" means each count gets only its own context's share, that needs a share per context, which the reading's counts don't keep. I recommend leaving it as built.
- Sun 15 Nov 6 PM PST holds if 1f97 starts by Sun 6 AM PST (INFERRED; it was waiting at 1:09 AM PST).

