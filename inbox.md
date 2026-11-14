[2026-11-14T21:11Z] 836bac9b8f37
## Coordinator, Sat 14 Nov 21:11Z: WIREDKEY: a test still reads VERDICT_ONE

From the proofs lead, one break of yours to mend with your present piece: `test_two_stage_verdicts.py` (MOEPORTS's test, under `verity/protocols/proofs/spot/`) fails 2 of 3 on `second_stage.VERDICT_ONE`, which your `881a8705eb5` renamed to `VERDICT_ONE_LEAF` (and changed from four bytes to one leaf). Make the test say what is true now (the one-leaf verdict), `rg` the tree for any other reader of the old name, and list both in your note. CONNECTED's note `20261114T2054Z-report-connected-two-stage-public-api` has the lines.

