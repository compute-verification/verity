[2026-11-13T10:19Z] 98c72380c3b8
compute-accounting answered on PRF-582 (2:05 AM PST, read by me in the issue): **the no-call window is enough to list `drawWithheld` now**, as its docstring says ("a window of no call, crediting nothing"), cap 182 to 181, lemma owner proofs. It shows `credited` accepts under a derived key, and it does not show crediting. **So the one-call witness is owed, not optional, and they ask us to size it with its date on the issue.**

In your next entry, a section "The one-call witness, sized":
- what it is made of (honest `callsRoot` rows and one round's receipts read back, their words), which of it exists already in your bridge's ten files and which is new;
- its size in files, lemmas and tries, and what could make it fail (name the one step you trust least);
- whether it needs a kept run with one call, and which run that would be;
- the day you would have it compiled, given the carry after chain 2 comes first.

**By 11:30Z.** I post the size and the date on PRF-582 by 4:00 AM PST (12:00Z). Until it lands their lines say "accepts, credits nothing shown"; write the same words in `drawWithheld_honest`'s docstring if they are not there.

