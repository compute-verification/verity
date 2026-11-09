[2026-11-09T00:31Z] da04701803eb
Coordinator 00:36Z: take (a). Change PooledLeakCarried in place. Its unwitnessed entry was never on main or checked, so regenerating it at the final statement is not a re-key of a landed entry; say so in the PR body. Note the reset rule is now PUBLIC (genesis at a first run, and after the verifier refused w-L), plus at most one prover-chosen reset a week. So #Cs is about 1,009 a week, not 2^m with a free bit per run. Charge exactly that.

