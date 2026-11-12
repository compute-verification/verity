[2026-11-12T21:14Z] 63c36df09ee3
training to BACKWARD2, Thu 12 Nov 1:14 PM PST. **TILES's review of #9417 at 84e7fc9161 is in: NOT READY, three must-fixes. Its whole text is the note `training-20261112T2025Z-report-TILES-review-9417` (lanes/training, synced): read it whole before you change a line; what follows is my ruling on each item, not a substitute.**

MUST FIX, each as the review says:

1. **The dP sum is FMUL then FFMA**, not FMUL and FADD (the SASS of `dot_do_o<false>` in the capture's cubin: per row 1 FMUL, then 7 FFMA, each into the last). `acc = F32MulV2(do_f, o_f) if acc is None else F32FmaV2(do_f, o_f, acc)`; the docstring says "FMUL, then FFMA"; a test on rows with subnormal products, which the FADD chain fails.
2. **A dQ part's add into its buffer flushes to zero** (the main kernel's 16 atomics are `REDG.E.ADD.F32.FTZ.RN`; the convert kernel's FADDs are plain). `P.F32AddFtzV2` there, `F32AddV2` for the adds of the buffers; the docstring names the atomics as the one flush-to-zero; a test with subnormal parts.
3. **`CONFORMANCE` says no more than a check has shown.** Narrow the string now to what the tests hold. When the every-word run has passed AT THE FIXED HEAD (musts 1 and 2 change Pre and Dq, so the run in flight at the old head does not count), put its run record in `evidence.json` and widen the string to it, as the forward does. Do not record "the fixes are the identity on every captured word" in its place: rerun.

SHOULD FIX, all yes, in this PR:

4, 5. The head's query-block order and a block's k16 order each pinned by a test with its negative control (scaled rows; the generator's head whose last key block tells both orders).
7. The T 16, D 32, BM = BN = 16 stand-in held by a `monkeypatch.setitem` test.
8. `NS=min(-(-SMS // (B * H)), T // blocks[1])`, so H 8 and H 12 are one id; say in the body that no word changes.
9. "and dV's" out of docstring 40.
10. The fixture module imports `BN` and `SMS`.
6. **The one-card probe with subnormal dQ parts and subnormal products: yes, and it is the one that needs a card.** Tell me by 2:30 PM PST which node you would run it on and for how long; launch nothing on a GPU before my word (a node stops tonight, and I ask infra if none is in reach). If no card is in reach before the PR is otherwise ready, the body says musts 1 and 2 rest on the SASS of the captured cubin and that the probe is owed, with the issue it is owed on.

Also: the body gains the two facts it lacks (FFMA in the dP sum, FTZ in the atomics), and the mutations the review ran that no test caught (M1, M6, M7, M9) are each caught at your new head: say which test catches each.

**Dates (PST).** Musts 1 to 3 and shoulds 8 to 10 pushed with the suite's and circuit-check's last lines: today 6 PM, as your commit was already due. Shoulds 4, 5 and 7 and the every-word run at the fixed head: Fri 13 Nov 10 AM, with the body final. The fixed head gets one more read by a reader I name; send me its SHA.

