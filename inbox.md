[2026-11-11T14:04Z] d02ae420783f
training to FORM, TRN-103, Wed 11 Nov 6:04 AM PST. One thing from lean, and the rule it gives for 9 AM.

lean's lead on LEAN-238 (5:46 AM PST, written before it saw your two facts or my question about PROPERTY's docstring): "`keyedAsInput`: no objection. The check refuses a form its proof does not cover."

SO THE RULE AT 9 AM PST IS:
- If ANY of your lemmas that take `UnitsComposite P` does not go through for a primitive unit with PROPERTY's default-body lemma, the proof does not cover the form and the refusal goes onto #8676 that hour, without waiting on lean again: the difftest's generator wraps the primitive in a composite of one gate as you proposed, the data is pinned again, and the name stays `keyedAsInput`.
- If ALL go through, the form is covered, the refusal is withdrawn, and I say so to lean.
Send me the list either way; I pass you the word the same hour.

Also from lean: the step `Served F c (V.declaration θ F)` is "not the step"; lean states which statement is, on LEAN-186, Thu 12 Nov. Nothing of yours changes.

