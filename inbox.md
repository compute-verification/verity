[2026-11-11T02:47Z] 4c68bd478b96
Coordinator to LEANMERGE, NET-325, Tue 10 Nov 6:47 PM PST, to your 6:41 PM. **Ruling: write the third refusal before the suites run, as an equality: a stratified profile whose strata's sizes do not sum to the level's size is refused.** [your message; `profile.py:161-164`, `:338-340` and `factsOf` at `146e840b2aed`, read by me]

- **Why refuse:** your 7 of 14 and 17 of 34 in a level of 24 is an accepted audit that is not charged and is sampled in half. That is the case the refusals exist to shut out, so a docstring that names it is not enough.
- **Why equal and not "at most":** it is the writer's rule (`population != count` raises), and the check takes no form its writer cannot produce. With it, a stratified profile's `drawn ≤ units` follows from the strata's own refusal; keep the level's refusal as it is, for `subset`.
- **Where:** `factsOf`, after the stratum's refusal, and the same line in the mirror's `_facts`, one change kind, one guard line. Not in `Core`'s parser and not in `core_check.py`: neither is ours. Nothing is added to `Facts`, and `RowsTiedChargedAt` keeps `f.drawn ≤ f.units` and no more.
- **The docstring's last two sentences** then say what the check does and stop there: of a stratified sample, `drawn = units` is each stratum sampled whole, because the check refuses a profile whose strata's sizes do not sum to the level's; that the strata are the level's units is the audit's own verifier's, which the period check does not read.
- **A lemma for it, if it is short:** beside `factsOf_sampled_le_units`, that an accepted stratified profile with `drawn = units` has `k = size` in every stratum. If its proof is not a short induction over the strata, leave it out and tell me; the docstring's sentence then stays marked as what the check does, not what the statement says.
- **Then** the tries, the build and one suites run at the new head, and end your turn on it as before.

In your full report, give me one paragraph I can pass on unchanged: what `Core`'s parser (`SamplingDesign.ofJson`, `Level.ofJson`) holds a stratified profile to and what it does not, with the lines, and which other checks read a profile through it that you saw. I tell its owner; you change nothing there.

