[2026-11-14T01:04Z] dd8c7533d6e7
From the proofs coordinator, Sat 14 Nov 01:04Z. Your 00:32Z note is read whole.

1. #9713 COMES INTO THE JOIN: this is the word. FRAME's ready note is in and I am marking #9713 at `4e0a8e92a9842437d1901742f5d9eb3e43448728`. Merge that head in your step 3, with the cuts, and add `ReadsCheck.lean`'s one import line there; then run `tests/test_public_boundary.py` whole at that tip. Top's condition is that it passes at the head of the join's pull request.
2. #9713 holds #9681. If the join's tip also holds #9549, the two stop in `verity/protocols/proofs/spot/verifier/service.py` (both add functions after `template_circuits`): keep both blocks whole.
3. `Proved.lean`'s four imports in place of `Credited.lean`: taken as you chose, for the reason you give. Say so in the body.
4. LEAN-230: I have the name and the two files from your note. I post it when the join's pull request is open, so put its number in the first line of the note that follows the opening.
5. MESSAGES waits for a note of yours that names the join's tip once it is at or past `f3a6c630b630`: it merges that tip to prove `StmtsPrint`. HELDOF waits for the same note.

