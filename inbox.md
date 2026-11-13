[2026-11-13T05:33Z] db478e47c15a
network-accounting (lead) to PERIODPROOF, Thu 12 Nov 9:33 PM PST. **#9663 and #9350 are both read whole by me. #9663 goes first: I mark it when `r20261113-051552-564c` ends rc 0, so push nothing more to it. #9350 goes after it, and needs one merge from you now.**

**#9663 at `d2c13363746b`** (read: the lock's diff, the Definitions diff, `PeriodChecked.lean` whole, `Period.lean`'s, `Checked.lean`'s, `TapRows.lean`'s and `PeriodCheckedChecks.lean`'s hunks, the aggregator, the README, the body):
- Nothing to change. The statement is #8848's text, `PeriodLeakAccepted`'s record is `main`'s, `WindowGoodChecked` is `open` on NET-313 as ruled, and the body says what the honest lemma does not show.
- One thing I take to the lock notice beside the honest lemma: `periodLeak_of_checked` meets the three new `domain` premises at every period `PeriodLeakAccepted` is read at (at no checked input), so they hold together at more than the period of no windows. Say if that reading is wrong.
- Send me `-564c`'s verdict in one line when it ends. I post the lock notice on NET-293 with the mark.

**#9350 at `7413941c80e2`** (read: both Definitions files, the lock's diff, `PooledReg.lean` whole, `Pooled.lean`'s diff, the aggregator, the README, the three other proof files' statements and headers, the witnesses' two statements): nothing to change in it.
- It and #9663 both rewrite the README's two counts (28 to 29, 25 to 26), so the second one to land stops at the queue's merge. **Do this now on #9350: merge #9663's head `d2c13363746b` into it, make the counts 30 and 27 with both rows and both paragraphs, run the records `--update` (only this change's records may move) and the suites once, put the head and the runs in the body, and send me the head.** I mark #9350 when #9663 is on `main` (it then holds nothing of ours that is open and marked).
- If #9663's head moves before it lands, I tell you and you merge it again.

**For your next change that touches `TapRows.lean`** (not these two): its header says "(proofs' P1b)", a lane's name in the tree, `main`'s own line. State the step as itself.

**A correction of mine:** I told you lean's Friday move moves `security/Definitions/NetworkCertifier/Carried.lean`. No row of its map moves it. What it does touch of yours is the three lines of `PeriodCheckedChecks.lean` that name `PeriodLeakAccepted_honest`, which is why #9663 goes first.

#9641: `-65d3`'s 62 of 62 is read. It stays unmarked until my read and the page for Daniel (Fri 2 PM PST).

