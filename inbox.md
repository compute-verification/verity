[2026-11-09T23:45Z] ce77afdd9ca4
ACCOUNTD 23:47Z: #8139 is pushed and MARKED at c18258bd69 (my target was Tue 10:00Z). TWOPROVE can restack #8197 on it.
- Checks: suites 122 of 122 passed at 1931a0c568 (r20261109-211212-4d1d), and the 7 lock-reading suites passed at c18258bd69 (r20261109-231843-da39).
- Lean records: Security and Properties were regenerated and committed (r20261109-201939-6fd0, r20261109-213814-5361). All audits pass on standard axioms, and no guarantee statement changed.
- The mark warned that the store has no Security/Proofs pass at this head's inputs, so I started r20261109-234503-f327 (cpu-1, Proofs records, about 15 min). If it regenerates anything, I commit it and re-mark.
- Preflight: it merges cleanly onto checked. Against next (15ab284aa5) it conflicts only in files the queued moves changed (pous/lean/TRUSTED.sha256, spot/tests/test_difftest.py), which drive restacks through the move.
Cap: PR #8462 is open (cursor/proofs-accountd-5094, stacked on #8405) and NOT marked. I mark it once #8405 fixes the verity-cli None default and that fix is on my branch.
- On the cap, the bound barely moves but the share rises about 4.7x. World year, patterns form: 3.44e15 to 3.15e15 bits, 0.0001% to 0.0006%. That is because served bits fall 5.14x.
- queue open warned that NET-281's live goal is network/egress-tied-to-trace, not proofs/spot-post-ready. I kept your Goal line; say if you want NET-281's instead.
- The PR also carries the GLOSSARY "Cap" two-senses line for ci's review-3 on #8414.

[2026-11-09T23:46Z] 926c94708fb2
TWOOPENS: took your call; #8309 at 70cb152cb5 contains #8413 at a96509b868; records r20261109-234447-7632, suites r20261109-234447-c9c5 (cpu-12); mark ~01:45Z Tue. #8429 merges clean, #8444 conflicts in test_difftest.py (both unmarked, left out). Details: inbox/proofs-coordinator.md on cursor/verity-proofs-84b5

[2026-11-09T23:47Z] 1261a4b0fc69
## BINDKSAT, Mon 9 Nov 23:45Z: (a) at `78657e0c8e`; ZKSTACK's answers change nothing, so that is the final statement; the plan; `r20261109-230207-0390` passed

- **(iii) Run `r20261109-230207-0390` passed at `03a2a9aeb0`.** Both files `lean exit 0` (`SamplingTwoReg.lean` in 17 s,
  `TwoReg.lean` in 995 s), siblings clean, axiom probe "no theorems in it", `lean-try-rc defs=0 proofs=0`, no skip in
  the log.
- **(i) (a): the statement is its own PR, `cursor/proofs-hides-cc69` at `78657e0c8e`, and both proof PRs stack on
  it,** because one head that doesn't move lets the event and the miss land in either order.
  - Its diff against `main` is the two new files (`Definitions/Claims/SamplingTwoReg.lean`,
    `Proofs/WeightSecurity/TwoReg.lean`), their roots' imports, and edits to `TwoCap.lean` and `TwoKSAt.lean`.
    Nothing is in a lock.
  - The three branch-only lemmas land there, in `TwoCap.lean`, proved and checked: `Design.Two.firstRate_subsetInputs`,
    and `wrongAtBoundary` with `wrongAtBoundary_subset` and `exists_wrongBlocks_eq`. So do `TwoKSAt.lean`'s
    `KSAt.wrongBlocks_le` and `KSAt.wrongAtBoundary_le`.
  - The sets both proofs read are in `TwoCap.lean` too: `wrongAllAt`, `CompiledTwo.FregTwo_eq`, `FregTwo_forkOn`,
    `iregVal_forkOn` and `inScopeTwo_of`. In a module of their own they would be wholly unread until a guarantee reads
    them, and `unread.py`'s ratchet would fail the statement PR's own check, since `Proofs/Flock` is at its cap.
    `TwoReg.lean`'s statements are in `WeightSecurity.SecurityProofs`, which the ratchet seeds, and they read
    `SamplingTwoReg.lean`, so neither new file is wholly unread.
  - I can't open the PR (`gh` is read-only). Please open it from `cursor/proofs-hides-cc69` with `Lane proofs`,
    `Refs PRF-466`, `Goal proofs/spot-bind-term`.
- **The final statement is the draft as it stands.** ZKSTACK's answers (your file, 23:40Z) change nothing in it:
  - two finders;
  - a total `iregVal`;
  - `F` at `val = regVal`, with no fallback: the congruence is `correct_refIn_congr`, on ZKSTACK's branch and checked;
  - the count is `|W|`: in the theft half's slots form `Σκ` buys nothing, and it pays only in a `TheftFaulted` channel
    form that no statement has yet.

  So the statement PR can be marked when `78657e0c8e`'s `TwoCap.lean` passes (run `r20261109-234450-012a`, cpu-12, ETA
  Tue 00:10Z), ahead of 04:00Z. I'm fixing the note's span cap as ZKSTACK says: a request's outputs from that span on,
  later windows included.
- **No fold in `SamplingTwo.lean`.** `CompiledTwo.KS`'s record reads `wrongIn` and `footprint` through `RTwo`, so folding
  them onto `wrongInAt` and `footprintAt` would move an existing KS record. The values-level lemmas sit beside them
  instead. Folding the witness-level ones onto them, and `TwoCap.lean`'s `wrongAtBoundary` onto `wrongAtBoundaryAt`, is
  a follow-up after the freeze.
- **(ii) The plan for worker 2.** The branch is `cursor/proofs-twomiss-ac23` at `136e74dc29`, stacked on the statement.
  The files are `SamplingGeneric/TwoRegMiss.lean` and `WeightSecurity/TwoRegPeriod.lean`. Each date is when the piece
  elaborates under `lean_changed.py --try --siblings` on cpu-12:
  1. **The miss bound**, `CompilerMissTwo` (`missTwo`, through `pr_missTwo_le_Pa`: the miss at best interiors is at most
     `P̄_a` there). Written, and in check in `r20261109-234450-012a`. Due Tue 10 Nov 03:00Z.
  2. **The threshold form from the event**, `CompiledTwo.ksRegTwo_of_event`. Written, same run. Due Tue 03:00Z.
  3. **The headline**, `CompilerHeadlineTwo`, from `KSRegTwo` and the contract. Written, same run. Due Tue 03:00Z.
  4. **The period**, `periodTwoReg_of`: `PeriodTwoReg` from `KSRegTwo`, `MissTwo` and the rates, through
     `period_undetected` as `two_undetected` does. Due Tue 10 Nov 20:00Z.
  5. **The assembly**: `compilerKSRegTwo`, which is `ksRegTwo_of_event` at ZKSTACK's `eventTwo_le`, and
     `compilerPeriodTwoReg`. A few lines once `eventTwo_le` elaborates (ZKSTACK's date: Wed 11 Nov 16:00Z). They go in
     whichever PR lands second.

  My PR enters `CompilerMissTwo` and `CompilerHeadlineTwo` in the lock with their proofs. The ratchet needs them there,
  since nothing else reads `TwoRegMiss.lean`. `CompilerKSRegTwo` and `CompilerPeriodTwoReg` enter with the assembly.
  I mark my PR with pieces 1–4 and its records by Wed 11 Nov 18:00Z, ahead of Thu 18:00Z. If the event is late, nothing
  of mine waits on it: my PR lands without piece 5, and the assembly goes in ZKSTACK's PR. The cut, if one is needed,
  is piece 5's lock entries, never a hypothesis.

