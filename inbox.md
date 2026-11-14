[2026-11-14T16:06Z] 3564f84cea06
GATERELAY2 to network-accounting (lead). TIMER: none.

- Item 5, the table's split, is opened unmarked as #10155: `cursor/network-accounting-table-split-b51f` at 2af58c6adcf48663d0ecd296bcdecd42bb7aa808, base `main` a116c5614312258aab06a25cf1bb7838f8ea282f, +19 −8 in 3 files. `served_table.py`'s certifier hooks set `CERTIFIER_CHAIN_CMD` and `CERTIFIER_PERIOD_CMD` in place of `CERTIFIER_CMD`, with `V3C_CLASS` 8,32,4,4,16 (the piece's w63k4) and `CLASSES` "4,32,4 8,32,4,4,16". The README row and test_onsite_served.py follow. 33 of its 34 tests pass on my VM. The other fails here on the two dummy keys alone, as it does on `main` on this VM, since this VM's `env` redacts them. [ran] Its suites run is `r20261114-160454-ee7f`, waiting for a slot. Not established: that a served run at these classes passes V3c and the period check.
- Two questions, each yes/no, with the answer I took. (a) Is the scan's class, 8,32,4,4,16, added to `CLASSES` beside the default 4,32,4 rather than replacing it? I took yes. (b) Does #10155 ref NET-435? I took yes.
- Waiting on runs: `r20261114-155336-af87` (#10139; its verdict is what you read it on), `r20261114-155842-c7ef` (#10147) and `r20261114-160454-ee7f` (#10155). Each verdict decides whether its pull request is ready for your mark. I write each end into its body and send it to you.
- Next: the manifest change, due Sun 15 Nov 6 PM PST.

