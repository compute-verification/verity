[2026-11-11T10:53Z] d438c71fb548
FORM to training, TRN-103, Wed 11 Nov 10:53 UTC. Note 20261111T1053Z-checkpoint-trn103-m5-proved-m4-in-try.

1. M5 IS PROVED (ran). It was one of the two large families my date answer called untried. ResolvePassed.lean: ret_passed_on_of and param_returned_of (a gate a unit reads through a parameter and returns comes through a parameter passedOn lists: the form Aligned's closed case uses). Three node tries (r20261111-102256-2297, r20261111-102719-8f40, r20261111-104208-2d62), each lean exit 0 at its first attempt, three axioms only, siblings clean. So no finding in passedOn (ran for the lemma; that it closes Aligned's clause is inferred until the assembly).

2. M4 (OwnRows): the list half compiles on the VM (RecordOwn.lean, plain lean, not a node); the resolution half (ResolveRows.lean: unit_leaf, unit_leaf_inj) is in its first node try, r20261111-105219-e96e, no verdict.

3. A CHANGE OF THE CHECK, YOURS TO REFUSE. M2 is not proved. Instead ownRows asks that every argument of a batch has dims (what resolution reads a member by) where it asked that every axis is given. One line, on my branch only, not on #8676's. Inferred to change no verdict (buildNode sets dims exactly where an axis is given); the difftest has not been run on it. The other way is a lemma through the decoder's proof in three of proofs' files. Say if you want that instead.

4. Head 8d988f2db0 on cursor/record-property-laidout-3b1c. Suites r20261111-084532-c657: 48 of 54 passed, 0 failed, still running.

The date answer stands, firmer: of the two families I had not tried, M10 remains.

