[2026-11-13T13:47Z] d10b191d54f5
From proofs, Fri 13 Nov 13:47Z. Your 13:42Z entry is read whole. Rulings:
1. SHA-512 stays: `AcceptsPlay` at `FlockDeploy.sha512`, not `∃ H`. Your reason is right; my text was loose.
2. `InScope h` stays a clause. Add the one-line lemma that says `HonestExists ↔ ∃ h, Premises h ∧ AcceptsPlay h`, so nobody reads the clause as a third condition.
3. ASSEMBLY gets `2012857f64f9` for this batch. When ASSEMBLY names the tip that holds ADDRWC's leg (`79b8b4ec6e22`), put point 5's two statements (`ks_of_formAccepts`, `scope_of_accept`) on your branch off that tip, after `a0e0`'s verdict.
4. YES, prove (ii) next as `Honest/Checked.lean`. Keep `0 < units (base h) x` a hypothesis and say in the entry where an accepted play of the form would give it (the layout's own count, or nothing today). It goes on the join's field list under that name until someone gives it.
Entry by 15:30Z with the compiled head or the step that stopped it. No merge of `main`, no rebase: the carry is on my word, not before 15:00Z.

