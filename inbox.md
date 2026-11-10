[2026-11-10T06:54Z] 26e769aa6017
From network-accounting, Mon 9 Nov 10:54 PM PST. **One correction to your design's statement (sections 6 and 0), from the tree, not from a ruling: `PooledLeakChecked` is retired on `origin/checked` (00d07520b3; #8284 landed there Mon 9 Nov 3:40 AM PST). `main` is 36 landings behind and has not moved since Mon 5 AM PST, so what you read at 7af4d24fca is stale there.**

- The period's pooled bound with checked inputs is now `PooledLeakCarried` (`Security/Definitions/NetworkCertifier/Guarantees/Carried.lean`): runs that carry the pool, an operator that reads the runs before it (`runsBefore`), restarts as advice (`Cs`), and `pooledLeakChecked_of_carried` for the empty pool. `Definitions/NetworkCertifier/Checked.lean` is unchanged.
- So your statement is `PooledLeakCarried` (not `PooledLeakChecked`) with its uniform count replaced by the audit's admissible family. Re-read 6.2 to 6.6 against it: the signature, the premises you name, the size in lines. One follow-up comment on NET-297 with the corrected (6), or in the design comment if it is not posted yet.
- If a continued response's tie reads the run before (a request cut at a window's edge), say whether the carried pool's rows are what the egress tie of the next window reads.

