[2026-11-14T15:49Z] e364d0e35cae
WALKER, Sat 14 Nov, 7:49 AM PST.

**A's head is pushed: 91d8c6b7748d34db4a0935b37fc6474e48c95bf0 (cursor/trn145-random-point-refusal-914d), READCIRC's two fixes on 9767b09f, checked by me first.**

- RAN: READCIRC's two tests, written by me from its description as `test_every_mac_reading_a_zero_row_is_left_out[xzero|anti]`, fail at 9767b09f (N 48 and 24, against 48 and 96 without the point) and pass with the patch.
- RAN on the VM at this head: `suites.py circuit_properties benchmarks/training_slowdown --changed 3d62f5507b01 --quick --fresh` gave `2 suites: 2 passed` (circuit-properties 143, training_slowdown 683, with `test_value_reading.py` and `test_program_reading.py` whole). No pinned reading moves. sdpa 3,776, relu 4,800, after_mask 4,096, rope 3,328 and lf 9,472 are unchanged. `tests/test_public_boundary.py` passed 56, and ruff is clean.
- READ, the change: `Tally.share` = min(1, the sum of the operands' zero shares); zero rows are recorded before the fewer-than-two-rows return; `points.py` 13-14 and 33 (and `rows`'s docstring) say "not zero"; the refused example at the second point is O1 - O1 + 1; the README holds the one-sentence over-leave note.
- Next: merge into C, then the quick runs and s-t20 at C.

