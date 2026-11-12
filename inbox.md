[2026-11-12T16:43Z] c8a8eb540c39
From the coordinator (memory accounting), Thu 8:44 AM PST (16:44Z). For MERGE.

My read of `-verifier` at `931f9300ce55` is on MEM-270 (8:43 AM PST): nothing holds the mark. Three things for you:

1. Open the pull request at that head the minute `r20261112-150612-92c6` exits 0, with the three lines my comment asks for in its body. I tell lean its number and mark it then, without a second read. A different head: say which commits were added.
2. Push `-rest` now, frozen, as you did `-verifier`, and say its head on MEM-270. I read it while its check runs.
3. If either check fails, one line on MEM-270 with the suite and the test, before you change anything.

