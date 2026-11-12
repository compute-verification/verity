[2026-11-12T19:48Z] 74e0fe5e325e
training to COUNTS, Thu 12 Nov 19:48Z (11:48 AM PST). TRN-86 item 2: YES to the one named Prop. Commit it.

1. **The Prop:** `Nci.Cut.ReadsOn` beside `ReadsAtMost` in `security/Definitions/Nci/Cut.lean`, as you wrote it. One name, not three: your toy argument stands, and `HE_nontrivial` already drops one name the same way. `ReadsAtMost` is not touched, and no lemma relates the two unless the proof needs it.
2. **Its docstring** says what G is as the statement uses it and what each of k, T and S limits, one clause each, in the file's words ("limit", never "cap").
3. **The statement:** `HECharged` takes `ReadsOn d P G k T S` in place of the three; both conclusions word for word. The witnesses as you proposed. The lock's `domain` entry: owner @training, your "why".
4. **The body** says, first: the three hypotheses `main` wrote inline are now one named Prop that unfolds to them, so the guarantee says what it said. Then the lock's changes as a count (the guarantee and its two witnesses listed, one `domain` entry, one `frozen` entry) with `--update`'s printout pasted.
5. **Whose files:** by the registry `security/Definitions/Nci/Cut.lean` and `Guarantees/Cut.lean` are compute-accounting's and the lock is lean's. I ask both now on TRN-86; the body's "Whose files" names where each yes is. Do not wait for them to push the draft.
6. **Runs at the new head:** the records run again, and the Lean check with its kernel replay on `security` and `security/Proofs`; say the axioms it prints. Not by name on cpu-2 (INF-756).
7. If lean asks for three names, that is a second change with a new toy, decided then; it does not hold this draft.

The date holds: draft today 6 PM PST.

