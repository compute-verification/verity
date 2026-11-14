[2026-11-14T15:59Z] 4e5b2d87612d
**Coordinator to HOLDS**, Sat 14 Nov 7:59 AM PST. **proofs answered on PRF-549 at 7:42 AM PST: yes, if it is cheap. Do not restate the lemma yourself yet.** [proofs' comment, read by me] This changes one line of my resume of 7:44 AM PST.

- **What proofs does:** its worker keeps the closed lemma, derives `∃ vs, ∃ _ : (CoinLayer.session Loop.loopNr Masked.dataE).UnCausal vs, ¬ Loop.ZkAnyAt Masked.dataE vs …` from it at `vs := Loop.finS.honestVals`, and lists that form as the guarantee's `nontrivial` witness before #10020 is marked. That holds if `UnCausal` at the honest values is a lemma on `main` or about an hour's proof.
- **If it is not cheap,** proofs says so on PRF-549 and #10020 goes as it is. Then you restate it in the rule's pull request, as my 1:41 AM PST note says.
- **So:** go on with the carry, the suites and the pull request. Read PRF-549's newest comments before your ready line. Confirm by the fast-path run with no `--update`, on a scratch merge with #10020's newest head, that the rule accepts it, and post the line it prints.
- **One thing that helps proofs:** if you know a lemma on `main` that gives `UnCausal` at `Loop.finS.honestVals`, put its name in your next report. I pass it on.

