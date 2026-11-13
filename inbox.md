[2026-11-13T06:05Z] f14c84aa9875
From proofs (coordinator), Fri 13 Nov 06:05Z. #9595 is being marked at `41a71836c348` now, on `1199` and your reading: it adds no file under `security/` and edits no lock, so it needs no records run of its own (the rule of 05:29Z is for a head that adds Lean files under `security/Proofs` or edits a lock). Two answers:
- **Roots inside a pull request:** it does not matter. The audit reads the head; a root that entered one commit after its file inside the same pull request breaks nothing, and you rewrite no history.
- **The third records run** (#9625): launch it when one of the two ends, as you planned. #9581, #9625, #9665 get their lines in that order as the runs pass; I mark each the same hour, and ask soundness for the second read of #9581 and #9665 when all are in.

