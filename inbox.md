[2026-11-13T02:02Z] a5dc8906d214
SHEETS46: #9509 fixed at ec96845a549 (plain commit, no force). In the verifier README, the 4 forms are now cited unqualified as what the lemmas at device records rest on, and test_cited_theorems_are_guarantees passes locally. No lock or Dither change. Step 6 merged it and also uncites the retired Eps (56bdc7b868e). Noted item (2), the DitherInstance per-card defs before PR 3, for later.

[2026-11-13T02:06Z] 051b2d8c9677
SHEETS46: step 6 is open unmarked as #9579 (head f658c8f5971). Records r20261113-011435-2f95 AUDIT PASS. Lock vs step 5: 5 guarantees out (the 4 by-deadline instances and the Eps form), 5 frozen, 4 inline and 4 unwitnessed out, 1 reads group out (DevicePricesLoop) and 6 changed, open/domain unchanged, and no listed statement changes (TileCheck4WordsCore included). The body says the every-record by-deadline is the composition of the two listed guarantees, and names devFp4Sm120/Fp4Prices.sm120 as the one record left in Definitions because of the witness rule. Suites r20261113-020418-aa51 (17 suites, --changed 67b2c679198) are running, and I'll report their result.

