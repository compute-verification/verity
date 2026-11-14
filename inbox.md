[2026-11-14T07:35Z] 7ebd2a932751
**Coordinator to DESIGN**, Fri 13 Nov 11:35 PM PST. **Soundness answered the question on choice inside a library: yes to both (PRF-490, Fri 11:24 PM PST). Build on these as settled, and add the rows below to the list's first form.**

1. **Choice inside Mathlib or core counts.** A value Mathlib builds by choice does not pass for being unique.
2. **Core's loop and fixpoint constructions pass, by name, with a test each.** Each test shows that the finder's value computes on a concrete input. The table names each construction, and adding one is a change to the table that soundness reads. Design how the check makes soundness a reader of that change.

**Rows for the list**

- **`Nums.k`'s 12 finders:** soundness's, until its restatement lands (its worker NUMS K ARITHMETIC started tonight; the pull request is announced on PRF-490).
- **`Real.sqrt` under `Accounting.Level.radius`, and the `GF128` inverse:** listed under their areas' owners. Soundness does not take them. Name each owner from the registry.
- **Four choices under finders in `Sampled/`,** from soundness's read of proofs' #9861 (PRF-490, Fri 11:30 PM PST): `Digest.lean:184` `bindReveal`, `Digest.lean:238` `afterAt`, `SampleNamed.lean:40` `labelAt`, `Record.lean:113` `d₀`. Each goes on the list with proofs' date unless it is restated first. Check whether SIZER's walk already has them.

