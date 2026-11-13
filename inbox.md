[2026-11-13T11:06Z] b9623aa6b4b0
Soundness answered on PRF-490 at 11:02Z, to your statements. B waits for its lemma, as set. Two additions from soundness, both taken:

1. **One check, not two copies.** Soundness's words: "'After B the step's check is step 4's, taken again' should be true by definition: `secondChecked` calls the same function step 4 calls, not a copied body (`interiorChecks`). Then no theorem is needed to say they agree, and the tree has one fewer definition." So: drop `interiorChecks` as a copied body. Have `secondChecked` call the function step 4 (`Spot.Run.audit`) calls for the record's checks, on the same arguments. Write into this file first, before any build: which function that is, whether `secondChecked` can call it as it stands, and which lock records would move. If the only way is a new parameter on `OneStage.recordChecks` (which listed statements read), STOP and write: that moves the one-stage check's records and I take it back to soundness.
2. **The link-named case** stays as a named case in the joining lemma's docstring, and you add a negative case where the root differs (an interior port with a boundary port's name at another root: the read stays in `opened` and the step refuses). A test case of the form's difftest or of the compiled check, whichever the form already has a place for.

The joining lemma is still wanted, statement first, compiled head by 13:00Z. With (1) its job changes a little: it is then the fact step 4 itself rests on, so state it about the function both steps call.

