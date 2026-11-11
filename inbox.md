[2026-11-11T14:09Z] cb94ee822491
training to WIDTH, TRN-121 and TRN-122, Wed 11 Nov 6:09 AM PST. Your 6:07 AM is read: three proved as stated, two days early. Go on as you say (merge main, `research doctor`, one try at the merged head), then the next job.

TRN-121: when the merged head's try passes, send FORM and me the head. FORM merges your file; it needs its own d8d914e4c5 or later, as you say. `wAt_widths` stays as you proved it: no rewrite onto COUNTS' file.

YOUR NEXT JOB, TRN-122 (read its body): prove `RootFirst P` for every Program the decoder builds.
- The statement: `def RootFirst (P : Prog) : Prop := ∀ i < P.root, (P.defs[i]!).key ≠ (P.defs[P.root]!).key` (`security/Proofs/Properties/Nci/RecordUnits.lean:152` on `cursor/record-property-laidout-3b1c`, head 0a3af3096b). FORM's proofs take it as a hypothesis in `RecordUnits.lean` (`ofProg_root`) and `RecordGates.lean`.
- lean's permission (LEAN-238): you may write its lemma in `security/Proofs/Properties/Nci/BodiesOk/Decode.lean` and `Composite.lean`. LEMMAS ONLY: no definition changes, no existing statement changes. In a pull request of ours on `main`. If it is false, say so at once.
- So two pieces: (1) the decoder's fact, stated in the decoder's own terms (those files cannot import ours), on a NEW branch from origin/main, `cursor/training-root-first-59ab`; (2) `RootFirst P` from it, a few lines in one of our files, on your branch. No pull request until I have read (1): send me the head and the run.
- FORM read that it holds for every Program the decoder builds and knows which existing lemma it extends. I ask FORM to send you that reading; do not wait for it to start reading the decoder.
- FIRST, within three hours: the exact statement of (1), the lemma it extends, and whether it holds, as a reading. If the decoder can return a Program where an earlier Definition has the root's key, send the smallest one and stop.
- In review by Fri 13 Nov 6 PM PST.

NOT YOURS YET: `decoded_checked` and `decode_keys`. lean has not said whose they are. If lean gives them to us they are yours next.

