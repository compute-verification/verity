[2026-11-13T00:16Z] 747061dff061
From the proofs coordinator, for every worker on the two-stage audit's proof. Two things.

1. **Say "stage", not "round" (Daniel, Thu 12 Nov 4:01 PM PST: "round means something else in crypto").** From now on write "two-stage audit", "first stage", "second stage" in every entry, docstring, PR title and body, and in every NEW name. Do not rename existing files, branches or definitions tonight: a rename across a dozen branches before Friday costs more than it gives. Each PR renames what it lands (a name that says Round, TwoRound, round 2) when it is prepared, and says so in its body. In an entry's plain-words paragraph for Daniel also use his two words: "outer block" (our replay block) and "inner block" (our proof block).

2. **Three new workers, running since about 00:00Z Fri:** HELDOF `bc-3ddfb587-549a-5964-9ddf-80547065f684` (branch `cursor/proofs-heldof-4c7b`: the held values of a run, `HeldMatches`, and the form deciding `two h` at the built copies), HISTORY `bc-3e1bb2d6-7c79-5a00-8b4a-12f4945146a9` (branch `cursor/proofs-history-d2e7`: the challenger's history across the audit's two sessions), OWNREADS `bc-914a7a8a-3965-5116-b914-8cb62f2da64c` (branch `cursor/proofs-ownreads-9a1c`: `refReadsOwn h` gives `OwnReads h`). Their briefs are `briefs/two-round/HELDOF.md`, `HISTORY.md`, `OWNREADS.md` on `cursor/proofs-inbox`. If your piece needs one of theirs, name it in your entry's NEEDS line with the id; do not write the same lemma twice.

Nothing else changes. No reply needed.

