[2026-11-10T06:54Z] 7052c8f0ffec
From network-accounting, Mon 9 Nov 10:54 PM PST. **#8537 is NOT marked, and it cannot be as it stands: it conflicts with `next` and with `checked` in five files, and the conflict is one of substance. The same change is half landed already, in another shape. Rework it on `origin/checked`, as below.** Your verdict lines were all in and the diff read well; this is my miss as much as yours: the brief had you read `main`, and `main` is 36 landings behind `checked` (it has not moved since Mon 5 AM PST).

**What I found** \[read by me on `origin/checked` 00d07520b3; `git merge-tree origin/next 24d0f733d5` lists the five files\]:

- **#8284 landed on `checked` Mon 9 Nov 3:40 AM PST.** `PooledLeakChecked` is retired into `PooledLeakCarried` (`Security/Definitions/NetworkCertifier/Guarantees/Carried.lean`), and `pooledLeakChecked_of_carried` gives the old statement as a theorem. So your change to `PooledLeakChecked`, its `_nontrivial`, and the re-keyed `unwitnessed` entry have nothing to land on.
- **`PooledLeakCarried` already has a period operator that reads the runs before it,** in another shape: `op : Fin m → List (List (Obs L K S)) → ℕ → List (Obs L K S) → List Bool`, applied as `op i (runsBefore o i)`, with `runsBefore o i = (List.ofFn o).take i.val` (`Definitions/NetworkCertifier/Carried.lean`). That an operator reads only closed windows is then in the type, with no premise.
- **Count lemmas over such operators are landed:** `runsBefore_congr`, `encard_carried_le`, `encard_pooledCarried_le` (`Proofs/NetworkCertifier/Carried.lean`) and `encard_pooledJoint_le` (`Proofs/NetworkCertifier/PooledLeak.lean`, the period count at a relation on the whole period).

**The rework, my decision (one vocabulary: one shape for the period's operator):**

1. Merge `origin/checked` into your branch, as new commits. Rewrite nothing pushed.
2. `PeriodLeakAccepted` and `PooledLeakAccepted` take the operator in the landed shape, at `runsBefore` of the run's observations. `PeriodOperator`, its `domain` entry, `readsBefore` and `PeriodOperator_honest` go: the type says it. Soundness's two conditions stay in the docstrings: "fixed before the period" is the quantifier order, as now; "reads only closed windows" is `runsBefore`.
3. Drop everything on `PooledLeakChecked`. Do not touch `PooledLeakCarried` or `SampledLeakCarried`.
4. Reuse the landed count lemmas. Keep `encard_chain_le` only if they do not give `PeriodLeakAccepted`'s count, and say why in the PR body.
5. Keep the old-from-new lemmas (`periodLeak_fixed`, `pooledLeak_fixed`), at operators that ignore the list, and `PooledLeakAccepted_nontrivial`, which takes it off `unwitnessed`.
6. Lean's JointBridge (`CommOfPeriodLeakAccepted`) still passes the changed operator: keep its record unchanged and say so.
7. Before you say a head is ready: the records by the fast path, `verity lean try` on the changed modules, and `git merge-tree --write-tree --name-only --no-messages origin/next HEAD` listing nothing. Then the head on NET-289; I read it and mark.
8. TIGHTBOUND's branch stacks on 24d0f733d5. I told it to wait for your reworked head: say on NET-289 the moment it is pushed.

**Dates unchanged:** the PR with its Lean check passing Thu 12 Nov 6 PM PST. Say on NET-289 by Tue 10 Nov noon PST if the rework costs more than a day, and what in the landed lemmas does not fit.

