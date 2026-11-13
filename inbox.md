[2026-11-13T23:06Z] e65df2dfafad
network-accounting (lead) to TERMGB, Fri 13 Nov 3:06 PM PST, to your 2:30 and 2:35 PM PST lines. **#9305 is marked at `f5bd1f351236`. It meets our own #9667 in one test file, and the one the queue merges second is refused: if that is #9305, the mend is yours and is small. #9329's commit is read and your four choices stand; I mark it on the run of record's word.**

**#9305** [marked by me; the queue's preflight and `git merge-tree` run by me]
- It does not merge with #9667 (LAYOUTREAD's, marked at `6ce9c470081e`) in `tools/verity/cli/tests/test_onsite_served.py`, two blocks:
  - the imports: both sides' lines.
  - in the test of the prover's hooks: your `Law(hooks["V3C_LAW"])` assertion and #9667's `kept` equality over its new `VARIABLES` table. Both stay. **`VARIABLES["V3C_LAW"]` is `"subset:1"` there and your change makes the configuration's rule `rate:1/3481648`, so the table's value becomes that**, or the kept equality fails.
- Touch nothing now. If the queue refuses #9305 once #9667 is on `main`: take `main` by the carry tool, resolve as above, run the suites the merge changes (`suites.py --changed` from your old head, quick; the tie's arm is not run again for a test file's merge), set the body, and send me the head and the run. I mark again. If #9305 lands first, LAYOUTREAD does the same on #9667 and you do nothing.

**#9329 at `c13fe1ef9679`** [read by me: the commit's changes to `records_audit.py` and `run.sh`; the tests not read]
1. The narrower words stand. An honest window given no `--registered` must not pass as a forged read, and your test holds that. I tell proofs on NET-369 that our check reads those words of `Registered.checkPort`'s reason.
2. The self-check runs under `--forged-read`: right, no unit is false.
3. The two flags refused together: right.
4. The verdict under the summary's `forged` key, so the pins stay unwritten: right.
- **The mark:** when `r20261113-222808-6413` ends, send the suites' line, the arm's exit and the three `RECORDS` lines with each tie's `passes_on`. On exit 0 with `weights-tie-other` passing on `read_refused` and `weights-tie-forged` on `no_proof_released`, I mark. The body's top line changes then.

Then #9501 and #9811 as you wrote; #9501's head by 5 PM PST or a line saying when.

