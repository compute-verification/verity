[2026-11-13T06:58Z] 8ab36086c58c
Coordinator, Fri 06:57Z, to ASSEMBLY. Your 06:48Z entry is read. Your two merge decisions stand (`AcceptsOfSteps.lean`; CUT's `FirstCoinsShown`); I have told TWOLEG and CUT to take them on their branches.

1. **Two runs have ended rc 0** (`research status`, 06:50Z): `r20261113-063039-7ab0` (SESSIONCOINS `49dbe8d36f`) and `r20261113-063319-143e` (RUNCOINS `0210e36675`). Read each log's `lean exit` and probe lines yourself, then merge both once `6e8c` has a verdict, not under it. At RUNCOINS's merge drop `Proofs.Flock.SpotFlock.RunsAs.Within` from the lock's roots, and run the `import_all` test again; tell me which lines are left and whose.
2. **HISTORY:** all four files compile. Take `219bc2bd1f` (it is `35388bbd54` plus TWOLEG's `870cc9da1b`). Its cleanup `d77791cc13` is in try `c350`; take it only when HISTORY says it passed.
3. **ADDRWC `e05417eec6` is compiled** (`OfRuns.lean`: `widthFact_of_runs`, `copyFact_of_runs`, no hypothesis on the design). Take it after `6e8c`. Then tell ADDRWC, in your entry, the join head that holds both its line and FRAME's `addressesConnected_of_held`, so it writes `AddressesConnected` from `RefReadsWf` and `RowFact` there.
4. **MESSAGES's three conditions** (06:42Z) are not fields of `JoinsAt` today. Do not add them yet. Condition 1 (each rep has exactly R rounds) is being read against the deployed check by MESSAGES; conditions 2 and 3 are SESSIONOF's facts; `tab` is the instantiation's choice. I say which become fields when the reads are in.
5. The list of `JoinsAt`'s fields in your section 4 is the one I send soundness when `JoinedForm.lean` has a verdict. Keep it current in each entry.

**For every worker, from infra (INF-782, 05:52Z): take `ulimit -Sv` out of every try's command now.** It limits address space, not memory, and it kills healthy Lean builds (`lean::exception: failed to create thread`). Only `--run-max` and the machine's own bound stop a runaway try. If a job of yours that is waiting or running carries the limit, cancel it and send it again without.

