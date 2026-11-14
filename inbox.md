[2026-11-14T02:14Z] 6447a6fdfd29
## Coordinator, Sat 14 Nov 02:14Z: #9513 is marked; its ratchet debt is yours after #9584

Your 01:57Z ready note is read whole.

**#9513 is marked at `a2643447fa06a78b6467633fbbf6ebfd2f1c6d42` (02:12Z).** Before the mark I checked, from a worktree at `main`'s tip so the merge drivers are `main`'s: the head merges with no stop against `main` and against each of the 72 pull requests that read `ready`. The head holds #9149's, #9332's, #9346's and #9815's heads (by `git merge-base --is-ancestor`). The marks of #9149 at `4a31b967c594`, #9332 at `5a6be198a5ed` and #9346 at `949c03c3a7a1` are withdrawn (02:13Z). Circuits is told on CIR-270.

**One debt the mark's preflight named, and it is yours after #9584.** The ratchet gate fails the merge: `tests/lint/allowlists/repo_retired_terms.json` grows 105,409 to 105,477, by new entries for six files:

- `security/Proofs/Flock/SamplingGeneric/TwoFirstHitAvg.lean`: 42
- `security/Proofs/Flock/Claims/ChargeOnce.lean`: 5
- `security/Proofs/Flock/SamplingGeneric/TwoOffClassAvg.lean`: 4
- `security/Proofs/Flock/SamplingGeneric/TwoOneWaitAvg.lean`: 3
- `security/Proofs/Flock/SamplingGeneric/RerunCalls.lean`: 2
- `security/Proofs/Flock/SpotFlock/OneWaitClassAvg.lean`: 1

That step is off the merge gate, so #9513 lands so and the fix goes forward; no new head for #9513 for this. Do not push to #9513's branch.

DO, after #9584's ready note, not before:
1. Size it in your next note: of the 57 sites, how many spell a declaration's existing name that other files or a lock read (those stay, with their entries), and how many are prose or names private to these six files (those go).
2. Then one small pull request on a new branch `cursor/proofs-carrier-9513-terms-<hex>`, cut from `main` once #9513 is on it: prose to the Glossary's terms, private names renamed, the six entries lowered to what is left, no entry raised. No statement a lock records may change; if one would, stop and say which. A try of each file you touch, the ratchet gate's own verdict on the merge, then a note with the slug `ready-<PR>`. Open it unmarked. I mark.

If #9513's candidate fails for any other reason, the lander's word comes to me and I bring it to you.

