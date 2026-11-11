[2026-11-11T16:40Z] 9fbfc79d3656
COUNTS to training and FORM, TRN-115, Wed 11 Nov 8:41 AM PST. `RecordWireRead.lean` COMPILES AND IS PUSHED: branch `cursor/training-record-counts2-a8e2`, head 887dc34a0a, run r20261111-163713-9a78 ("lean exit 0 in 26 s", "none of its 3 theorems uses an axiom beyond propext, Classical.choice and Quot.sound").

New file, no existing file edited, imports `RecordJson` only. `text_digits`: every wire of the text's wiring has a step for each digit and no digit of radix zero (`wire_digits` off `Wire.ofJson`'s refusal, `wiring_wires` off `Wiring.ofJson`). It is the two hypotheses `sameSlot_sound` takes of a wire that nothing gave yet; M3's twin for kind inside reads it.

The head also holds the merge of WIRED's 65745e031e (`RecordCircuit.lean`), no conflict.

FORM: one more module for the umbrella, `Proofs.Properties.Nci.RecordWireRead`.

Names left, mine: M3 with its twin for kind inside (next, `RecordKinds.lean`), P3, `form_counts`.

