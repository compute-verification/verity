[2026-11-13T20:19Z] fb9ff824e970
Your 19:23Z and 20:03Z are read, Fri 13 Nov 20:19Z. The case is (b), docstring-only, on the remade fixture (K 64, N 4). No statement changes.

1. Carry `cursor/proofs-onsite-fixture-docs-dde5` onto `origin/main` (a merge of `main`, no rebase), run the retired-terms lint and the ratchet gate again there, and open the pull request: `Lane proofs`, `Refs PRF-615`. Unmarked until I write.
2. The body says, in this order: that no statement, definition or proof line changes and no lock record moves (the run `r20261113-183536-a7da`, a fast-path result at the base); which of `HonestFixture`'s four conjuncts the whole-session tests reproduce (`HonestRunAccepted`) and that the other three (`HonestWitSatisfies`, `NoCollAt`, `TopsAt`) have no test in the tree; that the fixture's challenge tree is sampled by the node's operating system; and that the old docstring named tests and modes that were never on `main`.
3. The lock's two entries are not in this pull request. I put the finding (three conjuncts untested) and the choice between an issue and none to soundness and lean on PRF-615 now; the hand text changes with our next change to that lock.
4. Send the pull request's number and head, and the ready line when the quick suites pass at the carried head (`suites.py --changed origin/main --quick`, a node you name).

