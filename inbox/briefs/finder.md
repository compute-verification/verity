# FINDER: more bits of the binding error by proof (Lean)

**Where the figure stands.** SPOT's binding error, modeled from the Lean's bounds against 2^80 SHA-512 compressions in one verification phase: 2^-95.87 on the partition `main` defines; 2^-99.87 with the wide Merkle leaf (approved today, being built by another worker, Linear PRF-561); 2^-100.29 to 2^-100.63 on a partition that is not built. The target is 2^-100 on what is served, with margin.

**What the morning's report found** (`~/.research/notes/lanes/proofs/20261111T1551Z-report-daniel-morning.md`, section 2, and the factor table in `inbox/proofs-coordinator.md` on `cursor/verity-proofs-84b5`, entries headed `EXTFORM (PRF-528)` of 15:31Z and before): counting the first term by classes of gates was the one factor slack by proof alone, and that lemma is on `main` (#8906). Three more, none started:
1. **A finder that stops a further run's extraction at two waits:** 6.7 bits on paper, new mathematics.
2. **The event restated so the two tables are not each bounded over the whole sample:** 1.00 bit; it changes the guarantee's statement.
3. **`waits` 105 in place of 210:** 1.00 bit; a parameter of the audit record.

Also open under the same goal (`proofs/bind-cost-theorem`, due Thu 12 Nov 10 AM PST): the binding finder's cost proved in Lean; the first-hit guarantee of pull request #8915 rests on expected-time collision resistance and on a bound on the chance of the bad event that Lean does not yet give.

**Jobs:** first entry by 19:30Z: read the factor table and the Lean it cites, and tell me for each of the three what exactly would be proved, where, how many bits at the served sizes [run the evaluator in the tree; say which], and what it needs; and where `proofs/bind-cost-theorem` stands on `main`. Then build item 1 if your reading says the 6.7 bits are real, else the largest that is; items 2 and 3 as small pull requests if they are small. Item 2 changes a guarantee's statement: send me the restated statement before you prove it.

**Deadlines (UTC):** 19:30Z first entry; Thu 12 Nov 02:00Z a checkpoint with the statement of item 1 compiled or the reason it fails; Thu 12 Nov 16:00Z the pull request's ready line.
