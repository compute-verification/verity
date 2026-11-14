[2026-11-14T17:16Z] a644b2d65c34
READCALC, Sat 14 Nov, 9:16 AM PST. #10166 delta read, 6f77b21d1c61..63b34910e0a1 (one commit, `periods.py` and `test_min_cut.py`).

**READY 63b34910e0a1bb0c1b4a0e446f7c03d37bf481a7: the must-fix and all four should-fixes are cured in full, and no figure moves. One should-fix is new: the row's "within δ's p_evade" is printed, not checked, where it is written.**

1. The five cures:
- MUST 1, RAN: every remote "not scaled" line reads "the premises it takes beyond `Nci.Guarantees.DampenerRemote`'s, whose other 9 it shares (14 in the lock's record)".
  - READ: 14 is `len(record)` and 9 is `len(record) − len(PERIOD_PREMISES)`, both computed. The row writes no "five".
  - READ: `PER_BLOCK_BOUND` is `Remote.CITES[0]`, which RAN gives as `Nci.Guarantees.DampenerRemote`.
- SHOULD 1, RAN:
  - With `if stray:` replaced by `if False:`, exactly `test_a_remote_period_row_refuses_a_premise_neither_its_list_nor_the_per_block_record_names` fails (22 pass).
  - With `Nci.Assumptions.MadeUp` added to a copy of the record's `open` list, `periods.py --scenario reference-remote-h200` stops with: "ValueError: the lock's record of Nci.Guarantees.DampenerRemotePeriod takes ['Nci.Assumptions.MadeUp'], which neither PERIOD_PREMISES nor the record of Nci.Guarantees.DampenerRemote names: add each to PERIOD_PREMISES with the terms it is on". Restored after.
- SHOULD 2, RAN: the three remote "Set by" lines gain "The kinds are the per-block cuts' citations; the period statement, `Nci.Guarantees.DampenerRemotePeriod`, is the "not scaled" line's." The kinds table is unchanged.
- SHOULD 3, READ: the body says one parameter, `regime.remote.window_s`, is added to `data/envelopes.toml`. True.
- SHOULD 4, READ: the body names `r20261114-170706-9238` at 63b34910e0a1 as pending.

2. The outputs, RAN at 6f77b21d and 63b3491 with seed 0:
- `rho.md`: identical. Every figure file: byte-identical.
- `periods.md`: 6 lines differ, the 3 "Set by" and the 3 "not scaled" lines of the remote rows. Each change is words only: no number is lost, and the only numbers added are 9 and 14.

3. The rest of the range:
- RAN: one commit, two files. `test_min_cut.py` 23, `test_envelopes.py` 39 and `test_params.py` 7 pass at the head. The other seven rho files, and the 389, are the quick tier's.
- READ: the body is true at the head.
- RAN: the PR's added lines hold no issue id, run id, date, `art:` or `note:`. The one figure is a test assertion on a computed value.

4. Two of the 8:25 AM order's questions:
- RAN: T = ⌈period/3600⌉ is 168, 720 and 8760 for 604800, 2592000 and 31536000 s.
- READ: `window_ks` is `Param("regime.remote.window_ks", 2 * window)`, with `window = float(profile.WINDOW_DELTA)`. RAN: `test_envelopes` holds `remote.window_ks == 2 * window`.
- RAN, `Detection.delta` wrapped during both scripts:
  - `run.py` calls it on a sampled regime 4 times, all at luck 8 with J = 1. The worst tail/p_evade is 5.1e-85, so the refusal trips on none.
  - `periods.py` never calls it on a sampled regime.
- READ: the one `except ValueError` (`envelopes.regime`) re-raises, so nothing hides a refusal.

MUST FIX: none.

SHOULD FIX:
1. `pooled` prints "the escape's tail … within δ's `p_evade`" from `d.tail()` and never compares the two. `periods.py` reaches no `Detection.delta` on a sampled row (RAN).
   - Today no row is false: the registry's tail is 4.68e-97 against 9.09e-13. INFERRED: but a row at luck 1 with J 1000, inside both ranges, would print "within" over a tail above `p_evade`.
   - Refuse in `pooled` as `delta` does, or call `d.delta(ctx)` there.
   - This was in 6f77b21 too, and I missed it on the first read.

NOTE:
- READ: 9 equals the number of premises shared with `DampenerRemote` only because the stray refusal leaves none outside both lists. The new test holds that equality.

For the lead to read:
- RAN: no subscriptions or timers are held.
- RAN: the hook delivered PREMISES' message with the range; its account matches the diff.
- This VM was fresh again, so I rebuilt both worktrees.

