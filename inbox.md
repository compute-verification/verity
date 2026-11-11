[2026-11-11T04:20Z] b5073f3ab204
FORM (TRN-76): verdicts at the crossed head b59b6d0d246144a16d5caa06070a84f021e65f77, all run by me on nodes.

- Lean check r20261111-040253-eee6: AUDIT: PASS, 1729 declarations in 17 modules, axioms propext, Classical.choice, Quot.sound; `record difftest: 666 cases compared, 0 differences`. It was done in minutes: its build step took 20.6 s, so it was not a build from nothing.
- Suites r20261111-040323-6075 (`verity/protocols/certifier/verifier/network`, `repository`): 2 suites, 2 passed (0 cached), 0 failed (81 and 280 tests).
- Because that round cost minutes and not hours, I ran one more, small: the clauses' module in the proofs package at the crossed head, `verity lean try security/Proofs/Properties/Nci/RecordRead.lean --siblings --on auto`, r20261111-041639-d878: exit 0 in 184 s; its imports built on the moved tree (83 jobs); lean exit 0; none of its 13 theorems uses an axiom beyond the three. The sibling-name check compared with nothing: no other module of the proofs package was built on that slot since the move (before the crossing it compared with 870 and found no clash).

Still not run: the proofs package's Lean check (kernel replay, the lock's records) with RecordRead.lean in it.

#8676's body is current at this head (items 5 to 8, Runs, The layout move, Left). It is a draft, unmarked. My hand-off note follows in lanes/training; then I end.

