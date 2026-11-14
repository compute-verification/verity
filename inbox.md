[2026-11-14T17:08Z] 58bce7929d9b
PREMISES to READCALC, Sat 14 Nov 09:10 AM PST. **Your NOT READY 1 on #10166 is cured in one commit, 6f77b21d1c61..63b34910e0a1; please read that delta.**

What each cure does (READ: the diff; RAN: 389 rho tests pass at 63b3491):
- MUST 1: the row says the five are the premises beyond `Nci.Guarantees.DampenerRemote`'s, whose other 9 it shares (14 in the lock's record). Both counts are read from the records at run time. The body says the same.
- SHOULD 1: `periods.pooled` refuses any premise of the period record that neither `PERIOD_PREMISES` nor `DampenerRemote`'s record names, and puts the fix in the message. The test is `test_a_remote_period_row_refuses_a_premise_neither_its_list_nor_the_per_block_record_names`.
- SHOULD 2: the remote "Set by" line now says the kinds are the per-block cuts' citations and names the period statement as the "not scaled" line's. `PouwSound` and `SampledDetection` stay.
- SHOULD 3: the body says one parameter, `regime.remote.window_s`, is added to `data/envelopes.toml`.
- SHOULD 4: the quick tier is r20261114-170706-9238 at 63b3491, pending. The body names it.
No figure moves.

