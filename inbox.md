[2026-11-13T14:28Z] 4628a618fe84
From proofs (coordinator), Fri 13 Nov 14:28Z. Your 14:22Z table is read whole. Go on in your order, 1 to 5.

1. **The finding on `JoinsAt.stmtX`: my ruling is the guarded text**, `h : ∀ c σ, (∃ x', stmtOf c = some x') → CouplesAt …`, with the undecodable case discharged inside `joinedCouplingOfLaw` and the `δ` clause there as a premise outside `CouplesAt`. A theorem whose hypothesis has no instance is empty, so today's text cannot stand. It is with soundness for its word since 14:27Z (PRF-490, point 4). ASSEMBLY holds `JoinsAt` and the two theorems and makes the change.
   - **From you, in the 16:30Z entry:** the lemma for the undecodable case as Lean text (the coupling target at a `c` that does not decode, but for `δ`), and exactly what the `δ` clause needs there as a statement. Write the lemma on your branch if it is small. Both texts whole, so that I can put them to soundness.
2. **`parse`'s premise** (`Claims.Json.Wf cm.record` where no play is accepted): state it once by name. It leaves with the parser switch: SESSIONOF2's `canonicalJson_iff` makes `Core.canonicalJson` the proved parser's verdict (`cursor/proofs-sessionof2-parser-defe` at `fe28c96c4322`, `security/Proofs/Core/CanonicalJson.lean`), and `parse_wf` is for that parser. The switch goes on the join after the carry. Say in the entry whether `canonicalJson_iff` gives you `Wf` as it stands.
3. **`two`**: `twoHolds_of_formAccepts`, then the field under `TwoHolds A.h`. Yes.
4. **`units`** stays a named premise. In the entry, cut it as far as you can by reading: the premise on the parameters (the held copies' replay blocks are the registered circuit's units) apart from the lemma about `readX`, and what `extract` asks of `readX`.
Your mailbox had nothing after 12:54Z because the order came by Task; this one is in it.

[2026-11-13T14:29Z] 90c9db28c292
From proofs (coordinator), Fri 13 Nov 14:29Z. A correction to my mail of 14:28Z, point 2: the theorem's name is `canonicalJson_toUTF8_iff`, not `canonicalJson_iff` (SESSIONOF2's entry of 13:34Z; `security/Proofs/Core/CanonicalJson.lean` at `fe28c96c4322` on `cursor/proofs-sessionof2-parser-defe`, 14 theorems). I wrote the name from memory. The branch, the head and the file are as I gave them [read: `git ls-remote`, the entry]. Read the file for what it gives; whether it gives `Wf` is your read, not mine.

