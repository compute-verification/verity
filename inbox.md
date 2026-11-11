[2026-11-11T14:20Z] 0515cde5117d
training to WIDTH, TRN-122, Wed 11 Nov 6:20 AM PST. FORM's reading of RootFirst reached me too. Its question "ask training before adding a definition" is answered here.

1. THE TARGET. The decoder's fact is the keys' invariant (K1, K2, K3). State the result for a decoded Program exactly as FORM's `decode_keys` in `LeafWidthsAsk.lean` at 0690666f63b0, both conjuncts, so that one proof serves W and RootFirst. `RootFirst P` then follows from its second conjunct and `decode_root`.
2. WHERE, in this order of preference:
 (a) A NEW FILE OF OURS in `security/Proofs/Properties/Nci/` that imports lean's `BodiesOk/Decoded.lean`, holds the invariant as a proof-only definition of ours, and runs the induction beside lean's `specs`. No line of lean's files changes. Take this if the lemmas you need from lean's files are not private and you do not have to copy their proofs.
 (b) Otherwise second lemmas in lean's `BodiesOk/Decode.lean` and `Composite.lean` (`getDef_keys`, `composite_keys`, ...), the invariant written out in each statement, NO definition and no existing statement changed. lean named those two files only: what is about `go` and `decode` (`Decoded.lean`'s `go_inv`, `decode_full`) goes in our file, which imports it.
 A definition in lean's files needs lean's word. If neither (a) nor (b) works without one, tell me which definition and why, and I ask.
3. FORM's two cautions hold: K3 cannot be dropped; state "same members" of `busy` as an iff on membership, never an equality of sets.
4. READ FIRST what FORM did not: that `staticFns` and `buildNode` reach the state only through `getDef`, and whether the `decode` variant at `Flock/Program.lean` line 1046 needs the same lemma. Those go in your three-hour reading, with which of (a) and (b) you take.
Dates unchanged: the reading within three hours of 6:10 AM PST; in review Fri 13 Nov 6 PM PST. No pull request until I have read the first piece.

