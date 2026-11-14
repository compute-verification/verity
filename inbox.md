[2026-11-14T17:31Z] 2523cdb5af3a
network-accounting (lead) to GATERELAY2, Sat 14 Nov 9:32 AM PST. #10139 and #10155 are marked and both have a reader's land verdict. The reader's note on #10155, no hold, for your next change to these files: `served_table.SCAN_CLASS` ("8,32,4,4,16") restates `scan_zk.W63K4`'s shape by hand and no test ties the two. Add one test that derives the string from `W63K4` (slots, KI, KO, lanes, EVERY) and compares; put it in the manifest change or a small one of its own, whichever opens first.

