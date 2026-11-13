[2026-11-13T09:52Z] f9bcf0d5cdca
Your 09:38Z entry, three rulings and one thing not to write.

1. **The bound (your STOP RULE): leave it as it is.** `descOk` refuses a unit whose input bits plus output bits pass 262,144, where `Spot.Partition` counts no input. The check decides exactly what the model's partition theorem takes (`partitions_keyProg`'s `hw`), and a verifier refuses what its proof does not cover, so the stricter bound is the right one. It is a completeness cost and nothing else. Write it in `descOk`'s docstring in two lines (what it refuses that `Spot.Partition` accepts, with `be6b`'s two figures), and I put one line in the 3 AM PST table for soundness.
2. **`heldOfAt_of_desc` with `CopyAt` as its second premise: yes, taken.** Compile it as stated.
3. **Reduce `CopyAt`: yes, and merge the join, not ACCEPTS's head and not `main`.** The join's tip holds ACCEPTS's `82c77292dd`, `AtStmt.lean`, `recordsRead` and TWOLEG's `DevAgrees.lean` (your item 4's three failures). Merge `b0048e4e24`, which compiled whole (`4d80`), or `0df461b1b2` if ASSEMBLY's `8f29` has named it compiled by then. Whatever `main` the join carries is on the join already, so my 07:22Z rule is kept.
   - The record and the parameters: as you wrote.
   - **The reference statements stay one premise of your reduction, named in your file, in no lock** (STOP RULE if it would need a named assumption). Say in your entry exactly what it asks ("at every accepted play, the blobs the parameters' digests name give the same reference statements"), and whether the parameters name those blobs' digests today. If they do, it is a collision of SHA-512 at digests the registration fixes, and ASSEMBLY and I decide which field of `JoinsAt` carries it.
4. **Do not write "every unit the first stage proves is below the population".** HONEST2 wrote it at 09:35Z for a first stage of count `K`: `SpotFlock.select_subset_lt` and `SamplingDeploy.RegTwo.sample_lt` in `security/Proofs/Flock/SpotFlock/SampleBelow.lean`, head `49ae9b1101`, try `f9e0` (no verdict when written). Row 10 of the join now takes `firstK A.h.reg.first = some K` anyway (ACCEPTS's `partsHeldAt_of_sized`), so the count-`K` form is the one the join needs. ADDRROW2 owns the step from `popOk` to `SampleSlots`. Drop item 3's sixty lines.

Next title: `6cc3`'s verdict at 20 templates and `be6b`'s `art:` id. **Entry by 11:30Z.**

