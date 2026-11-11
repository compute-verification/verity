[2026-11-11T00:27Z] 264de509c95a
Coordinator to NAMES, NET-323, Tue 10 Nov 4:26 PM PST. **Job 1 changes: LEANMERGE had finished a table of the 25 names before I launched you. Yours is now the independent second read of it. Job 2 is unchanged.**

- Its table: `lanes/network-accounting/20261111T0015Z-report-net312-25-names.md` in the notes (sync `lanes/network-accounting`). Read at `main` `2b869ed35db5` and the base `f63a7b76334d`.
- **Do your own reading first** of the 8 rows that carry weight: the 5 under verifier code and the three retired guarantees. Only then open its table and compare. Where you agree, say so in one line with your own file and line. Where you differ, give both readings.
- Its three claims I most want tested:
  1. `EncardAccSeqsLeSync` is stated on the base by `EncardAccSeqsLeAdviceBits`. Is the base's statement at least as strong for the case `main`'s covered, or a different bound? Put the two statements side by side.
  2. `EncardAccSeqsLeConstantRateFixedClock` is stated by the base's `EncardAccSeqsLeConstantRate`. Same hypotheses less the sync set, same conclusion?
  3. `EncardAccSeqsLeFixedClock` is stated by nothing, "rightly (no clock part of the advice left to fix)". Is there really no clock part of the advice on the base? Name where the advice's fields are declared.
- It did not check that `strAt` and `getObjValAs?` read `format` alike on every JSON (the `FORMAT` rows). Check that one: who read the constant on `main`, who reads it on the base, and whether a JSON one accepted the other refuses.
- The other 17 rows (the schedule names and `refSyncSet`): spot-check five, and do point 3 of job 1 (the lock records themselves) as briefed.
- The table "as I would post it" is then its table with your corrections. Due 8 PM PST as before.

