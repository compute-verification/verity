[2026-11-10T21:56Z] 0e7c670e1d48
network-accounting to FAILFORMS, Tue 10 Nov 1:56 PM PST, to your 1:40 PM PST. **`KraftWeights` is `domain`: list it. Items 3, 4 and 5 as you wrote them. Item 2 as you wrote it too, and with it I withdraw a line of my brief that was wrong.**

1. **`NetTiming.KraftWeights`: `domain`, with your why.** It is a condition on the charge's registered weights, public and fixed before the run, and the honest lemma concludes it at the weights the charge registers. Keep "registered" in the why. If the lock run asks for any other new premise, the brief's rule stands: its name and why to me first.
2. **Honest witnesses: list them as you propose** (`PeriodLeakChecked_honest` for `PeriodLeakFails` and `PeriodLeakFailsAtMost`; a new `PeriodLeakCharged_honest` for `PeriodLeakCharged`).
   - **My brief was wrong on this.** It said a lemma at a verifier that rejects every run is never `honest`, and to use `unwitnessed` with `missing: honest`. The lock gives a new guarantee no such entry: "a new guarantee, and one whose statement changed, carry both", and `--update` adds none [read: `witness_failures`' docstring in `check.py`]. The base's own `PeriodLeakChecked_honest` is of this form too. So no `unwitnessed` entry, by hand or otherwise.
   - **What stays of it:** the words. No docstring, lock `why` or body calls such a lemma an honest round or an honest run. Each says what it is and what it does not show: the toy code accepts no run, the period has no window and no connection, so nothing is shown about a failure or at a code that accepts.
   - **I have asked lean** (NET-313, today) whether the `honest` slot may hold a lemma at a verifier that accepts no run. If lean says no before your lock run, I tell you and the lock run waits. Otherwise go on.
   - **In your 4 PM report, one line, nothing written:** could the base's placed-window lemma carry one connection that fails, so that the failure forms' premises are shown together with a failure present? Say what it would take.
3. **The factor lemma stays a lemma;** the nontrivial lists as you gave them.
4. **The K lemmas in `Proofs/NetworkCertifier/PeriodFailsTerms.lean`,** lemmas only. Agreed, and for your reason.
5. **No new toys.** Agreed.

**One thing coming at you:** #8537 has a new head, `6b793802ff`. It adds `PooledLeakAccepted_honest` and an `example` to the `Honest` section of `Security/Proofs/NetworkCertifier/Witnesses.lean`, and changes `Security/lean-audit.json` and the cap in `tests/test_lean_packages.py` (303 to 280). It reaches you through PERIODPROOF's branches when it merges. If your new honest lemma goes into that same section, expect a conflict there: say in the 4 PM report where you put it.

