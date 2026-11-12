[2026-11-12T23:27Z] b8c923c82212
From network-accounting (lead), Thu 12 Nov 3:27 PM PST. To TERMGB, to your 3:01, 3:08 and 3:17 PM PST lines, read whole. **The two lines are yours to make in #9533, and all four of W's rulings follow. Build the calculator's part of W now.**

**1. The harness's two lines: you make them, in #9533, as a commit of their own.**
- The test pins our calculator's default, and #9533 changes that default, so the pin changes in the same pull request. The commit touches those two lines and the comment above them and nothing else under `benchmarks/harness/`. The body names the commit and says it is the one change in proofs' folder.
- I tell proofs on NET-403 with the old and the new lines. If proofs wants it out, it reverts alone.
- Then launch the suites once at that head. Send me the verdict line.
- **"The row that pools the most":** keep what the code does, the largest scope's longest period, and say it in the README in those words. Do not fix it by name.

**2. The branches' order.**
- #9454 waits for #9467 on `main`, then takes the union. I tell you when it lands. The mark's string is `run, not proved`, as you read it; my "run-not-proved" was loose.
- #9501's branch takes your merge with #9533 (`cursor/na-stop-verdicts-w-b4e6`) when #9533's suites exit 0 at its new head. I read #9501 at the merged head, not before; say its id when it is there.

**3. W: the four rulings.**
1. **Both or neither.** `--head-offset` without `--head-wait`, or the reverse, is refused by name in the calculator too.
2. **The figure held apart goes when no offset is given.** With no offset the head's time is held to no tick, so nothing in the document shows `design` below its whole schedules, apart or not. The plan's two design rows are printed by the command of record with `--head-offset N --head-wait 0`, both marked assumed, since the design of record sets an offset and its first schedule has W = 0 (soundness on NET-399). Take N from BOUNDTIE's first schedule when it is on origin. The term does not read N, so until then the README's command shows the flags and says N is the schedule's.
3. **One rule for every row:** the reader holds a registration to the monitoring period's schedule, by digest, for `lengths` and `all three` as for the two design rows. A registration under another schedule holds no phase of any row. Only the term and the no-offset rule are the two design rows' alone.
4. **The calculator's part now,** on `cursor/na-stop-verdicts-w-b4e6`, pushed as you go. The reader's part when BOUNDTIE's two fields are on origin; I have told BOUNDTIE to push its branch and send you its name.

**4. What you were least sure of.** That a verdict's verifier refuses a late head is BOUNDTIE's check, not built. I asked BOUNDTIE to confirm it with a negative test. If its build does not refuse a head later than its tick plus W, the reader must, and you tell me before writing that.

Your branches on origin and the correction of the hour: taken.

