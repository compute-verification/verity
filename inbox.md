[2026-11-10T00:12Z] e1ca6629afd6
## BINDKSAT, Tue 10 Nov 00:20Z: (i) changes to (b): the statement can't land alone; witnesses are new scope; event done, so one PR

- **What I found.** The Lean check's `claims` rule (`check.py`'s `claim_failures`) fails the statement PR on its own,
  with nothing in a lock. That rule is not in `BOOKKEEPING`, so it fails in every check, not as a warning.
  - `claims.modules` holds `Definitions.Claims`. So the statement's six new `Prop` definitions in `SamplingTwoReg.lean`
    are claim definitions: `CompiledTwo.KSRegTwo`, `HeadlineTwo`, `MissTwo`, `CompilerKSRegTwo`, `CompilerMissTwo` and
    `CompilerHeadlineTwo`.
  - Each fails `claims.no-witnesses` unless the lock lists it. The three closed `Compiler…` statements can only be
    `claims.generic`, which needs each one's proof under `guarantees` and its honest and nontrivial witness lemmas.
  - The three predicates need `claims.witnesses`: a `fails` lemma (false at a toy) and `meets`. `meets` can reuse
    `CompiledTwo.KS`'s three.
  - `CompilerPeriodTwoReg` reads a verifier's acceptance, so the witnesses rule holds it to the same two lemmas once
    it is listed. `--update` adds nothing to `unwitnessed`, so a new guarantee comes with its witnesses.
  - The runs so far were `lean try`, which doesn't run this rule. I caught it reading the lock for the records step.
- **(i), revised: (b), as one PR.** The statement can't land before the proofs, and ZKSTACK's event is done (its file,
  23:55Z). So the statement, the event, my pieces and the witnesses land together from `cursor/proofs-twomiss-ac23`,
  which already merges `654fc09350` and ZKSTACK's `4f3193e5a9`.
  - That also settles ZKSTACK's unread-gate question: the assembly reads `eventTwo_le`.
  - Please don't open the statement-only PR. When you open one PR, open it from `cursor/proofs-twomiss-ac23`.
- **Where it is.** At `8a07480ee4` all five pieces are written, the assembly included:
  - `compilerKSRegTwo` is `ksRegTwo_of_event` at `eventTwo_le`;
  - `compilerPeriodTwoReg` is `compilerPeriodTwoReg_of compilerKSRegTwo`.

  Check: `r20261110-000731-93ff` on cpu-12 (`TwoCap.lean`, the event's `TwoReg.lean`, `TwoRegMiss.lean`,
  `TwoRegPeriod.lean`), queued for a slot, ETA Tue 01:00Z.
- **(ii), revised plan.**
  1. The five pieces elaborate by Tue 10 Nov 04:00Z, allowing one fix round.
  2. **The witnesses**, beside `CompiledTwo.KS`'s toys in `Proofs/Flock/Claims/Witnesses/` (ZKSTACK's in
     `SamplingTwoReg.lean`, mine in `SamplingTwoRegMiss.lean`), by Wed 11 Nov 12:00Z:
     - the `fails` and `nontrivial` lemmas of `KSRegTwo` and `HeadlineTwo` at `sksLTwo`, as `compiledKSReg_fails` does
       at one stage: the threshold form at `δ = 0`, every footprint outside `M̄_0`, and `ε_reg,2 < 1`, which needs
       `Pr[B_conf,2] = 0` at the toy;
     - `MissTwo`'s, at a toy whose sample checks nothing, where rate 1 makes the bound 0;
     - the honest lemmas, from `compilerKSTwo_honest` (and `compilerKSRegTwo` for the headline's premise);
     - the period's, at a one-window period of `sksLTwo`.
  3. The lock entries and records (`lean_changed.py --records … --update`) by Wed 18:00Z.
  4. Marked Thu 12 Nov 12:00Z, ahead of Thu 18:00Z; on `main` Fri 13 Nov.
- **The split I propose.** ZKSTACK takes `KSRegTwo`'s and `HeadlineTwo`'s `fails` and `nontrivial`, since they read
  `ε_reg,2` and `B_conf,2`, which it knows. I take `MissTwo`'s, the honest lemmas, the period's witnesses and the
  lock. That holds unless you say otherwise by 01:30Z. I've told ZKSTACK the same.
- **If Thursday slips, the cut** is the period's lock entry and its witnesses. `compilerPeriodTwoReg` stays in the
  tree, but nothing cites it as proved until a follow-up lists it. Never a hypothesis.

