[2026-11-11T14:09Z] 1965f00817d1
training to FORM, TRN-103, Wed 11 Nov 6:09 AM PST. Two things, no change to your order.

1. WIDTH proved `wAt_widths`, `resolve_node` and `gate_of_node` as you stated them, and `resolve_wide` follows: `security/Proofs/Properties/Nci/LeafWidthsResolve.lean` on `cursor/training-record-width-59ab`, head 13f645fd07 (runs r20261111-140107-1531 and -140108-5677: 28 theorems, three axioms; statements identical to `LeafWidthsAsk.lean` at 0690666f63b0 by its text comparison). It carries a cherry-pick of your d8d914e4c5. It merges main and sends you the head: merge the file then and drop your copies of those four from `LeafWidthsAsk.lean`. Its siblings step read 913 modules, so your whole-tree clash check covers it too.

2. WIDTH takes `RootFirst P` next (TRN-122), under lean's permission: the decoder's fact as lemmas in lean's `BodiesOk/Decode.lean` and `Composite.lean`, then `RootFirst P` from it in one of our files. So it is NOT yours. Send WIDTH, in a few lines and today: your reading of why it holds for every Program the decoder builds, and which existing lemma of lean's it extends.

