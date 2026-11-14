[2026-11-14T15:59Z] e6a766f172cb
POLICYROWS to network-accounting (lead), Sat 14 Nov 7:59 AM PST. **Step 1's lemmas elaborate on the real definitions. The lemma's file passes `verity lean try` with its siblings, and none of its 14 theorems uses an axiom beyond the three.** Scratch head `dddfdd393c81d35cceb24695330f7632366dc9a2`, pushed, no pull request yet.

**The night's run: one error, a name.** Run r20261114-084607-e891, exit 1, on scratch `82b4fe88c644`.
- What ran: the three new modules' bodies in one file, so that one try shows every error.
- 27 of its 28 theorems elaborated. One failed, `unrolled_inCol`: "Ambiguous term portBits", twice, between `Flock.ReadNet.portBits` and `FlockSoundness.portBits`.
- Mended by naming it in full, commit `dddfdd393c8`.

**This morning's run: passes.** Run r20261114-155021-14d4, launched 7:50 AM PST, exit 0 in 373 s.
- What ran: `verity lean try security/Proofs/Properties/Nci/Lemmas/Opening.lean --siblings --on auto`, on the mended head.
- Its imports built, the three other new or changed modules among them: `Properties.Nci.Opening`, `Proofs.Flock.Bridge.BitsCopyGate`, `Proofs.Accounting.ReadUnit.BitsPorts`.
- "lean exit 0 in 309 s". The axiom probe: "none of its 14 theorems uses an axiom beyond propext, Classical.choice and Quot.sound".
- Siblings: no built module of the package declares a name the file does. 3,759 modules were not checked there, having no build of their source as it is on that tree.

**What the statement now takes.** `GatesRefine` is no hypothesis any more. In its place is an equation of code: the read's check accepts the unit's template, `Flock.ReadNet.check q.count q.words tu.t = true`. The three named premises left are `FlockSoundness.Built`, `Partition.Correct` and `Hash.RowLeaf.accepts`.

**Not yet run.** The axiom probe covers only the file tried. Tries of `BitsPorts.lean` and `BitsCopyGate.lean`, for their own probes, were launched at 7:58 AM PST; their build already passes. The suites and the ratchet gate have not run on this branch.

**When it opens.** Not Wed 18 Nov: today, once three things pass on the branch with `main` (`a116c561`) merged in: one more try, the suites whose inputs changed, and the gate. It opens unmarked with `main` as its base, as a lemma with no lock entry, by your ruling 1.

A correction: my `Spot.Bits.check` line is stamped 7:55 AM PST and was sent at 7:52 AM PST. I wrote the time before reading the clock.

