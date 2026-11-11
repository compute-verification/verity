[2026-11-11T14:49Z] cb2d3f0f1535
COUNTS to training (copy FORM), TRN-115, Wed 11 Nov 6:49 AM PST. Where my eight are used: what I found, and I stop it here for `decoded_checked`. (My note to WIDTH said 6:50; it was sent 6:44.)

RESULT: no statement of mine found short of what its caller needs. One caller exists, as stated, and it composes.

- THE CALLER: `resolve_wide`'s `hW` (WIDTH's, proved): for every composite, `body.retType = ret ∧ BodyWide body`. It comes from `decoded_checked` and `hD` (`DefOk` of every Definition) through `bodyBad_sound`: `hB` is `(hD d hd).2`, `hI` and the check's pass are `decoded_checked`'s second and third conjuncts, and `hret` (`ret.count = retType.leaves`) is `decoded_checked`'s first rewritten into `CalleeOk`'s `body.ret.count = ret.leaves`. A scratch theorem with exactly that statement compiles on a node: run r20261111-144204-8f83, lean exit 0, 2 theorems, the three axioms. It is not pushed and is in no file of the tree.
- A primitive: `hW` asks nothing of it, and `BodyWide default` holds by itself (the scratch's second theorem).
- The other seven (`wAt_mem_rangeWidths`, `homo_wAt`, `cover_spec`, `eqRange_sound`, `residues_mem`, `partOk_sound`, `seqBad_sound`) are called only by each other and by `bodyBad_sound`, inside my files.
- FORM's integration branch at 0a3af3096b called none of the eight: `LeafWide` is still a hypothesis there.

NOT READ: FORM's 590d323f4d (the unit of a primitive), and how `LeafWide` is to come from `resolve_wide` (`OperandsOwn`, `ReturnsOwn` give the resolved gate; whether `tyOf` at a row's leaf is the row's leaf width as `InLeaf`/`OutLeaf` state it, I did not check).

