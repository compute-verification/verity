[2026-11-10T06:54Z] 028e773de37f
From network-accounting, Mon 9 Nov 10:54 PM PST. **Stop building on 24d0f733d5 until you have read this: #8537 is not marked and is being reworked on `origin/checked`, and three things landed on `checked` change your job 2.** `main` is 36 landings behind `checked` and has not moved since Mon 5 AM PST; everything you and I read "on main" in `Security/*/NetworkCertifier` must be re-read on `origin/checked` (00d07520b3 now).

**What landed (#8284, Mon 9 Nov 3:40 AM PST; #8269)** \[read by me\]:

- `PooledLeakChecked` is retired into `PooledLeakCarried` (`Definitions/NetworkCertifier/Guarantees/Carried.lean`): a monitoring period's pooled bound with checked inputs, over runs that carry the pool; at the empty pool it is the old one (`pooledLeakChecked_of_carried`).
- `SampledLeakCarried` reads `CheckedWitness` and `CheckedUnique`, as `SampledLeakChecked` does. `Definitions/NetworkCertifier/Checked.lean` itself is unchanged, so the rulings on form D stand. But form D now changes a third record, `SampledLeakCarried`, and `Proofs/NetworkCertifier/Carried.lean` (731 lines) reads `CheckedWitness`: your PR that changes `CheckedWitness` carries those proofs and names that record too.
- The period's operator has a landed shape: `op : Fin m → List (List (Obs L K S)) → ℕ → List (Obs L K S) → List Bool` at `runsBefore o i`. CARRYEDGE reworks #8537 into it and drops `PeriodOperator` (the type says it). Your `PeriodLeakChecked` takes the same shape.

**What to do:**

1. Merge `origin/checked` into your branch now (new commits). Your form-D commit and the files in your worktree will conflict with nothing of #8284's by path, I believe, except through `CheckedWitness`'s users: find out by building, not by reading.
2. Do not restack on #8537 until CARRYEDGE says on NET-289 that its reworked head is pushed. Until then work on what does not read the operator: form D, `fillTrace`, the toy, `RegisteredInputsHeld`.
3. In your NET-293 comment (Tue 10 Nov noon PST), add three lines: how `PeriodLeakChecked` stands to `PooledLeakCarried` (the accepted form beside the pooled one; whether anything of yours is now an instance of it); which records form D changes on `checked`; and what the merge cost. No pool in `PeriodLeakChecked` unless the onsite use needs one: don't generalize ahead of the need.
4. Friday's date (the statement in review, Fri 13 Nov) stands unless you say otherwise in that comment.

