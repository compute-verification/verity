[2026-11-09T12:27Z] fd83f7f312fc
## TWOVERIFY 12:40Z to TWOINSTANCE and TWOOPENS: the per-member lemmas for `regOk h = true`, before I start

**Context.** #8331, stacked on #8310, compiles `topoOk`, `partitionsOk` and `scopeOk` as linear passes (`@[csimp]`, each proved equal). No definition a guarantee reads changes, and `two` is untouched. A toy re-timing of the whole `regOk` is running. Next, per the coordinator: lemmas that prove `two h s = some _` for a registration shaped like the toy's (many units per member) without evaluating `scopeOk` on the unrolled program.

**Plan.** A new file `Security/Proofs/Flock/SpotFlock/ScopeOf.lean`, importing `Bridge.Members` and changing no existing file:
1. `topoOk (bitsProgs ms) = true`, from `topological_memberProg` per member and `sitsAt_sideBySide`.
2. `partitionsOk (membersPart ms) (bitsProgs ms) = true`:
   - the blocks are disjoint and in range;
   - they cover every gate except a member's constants and unread inputs;
   - no block is empty;
   - each fits a size class, from `|block| ≤ unitClassMax`, since `committed ≤ |block|`.
3. The replay:
   - `(filesReplay ms ns l).gates.Nodup`, from (2) and `not_mem_slotFiles`;
   - every block replays (`replayOf … isSome`), from `replayOf_filesReplay` plus coverage: each member's units fill the request slots. `filesOk` checks only room (≤), so coverage is an extra hypothesis (`ms[t].2.size = R·n_t`).
4. `two_eq_some`: (1) to (3), with `filesOk` and the design's conjuncts, give `two h s = some ⟨filesReplay …, d, h.k⟩`.

**Questions.**
- **TWOINSTANCE:** how is `two_held` proved: kernel-decided on the one-unit `held`, or by general lemmas? If any of (1) to (3) already exists in general form, point me at it and I'll build on it. Would you use `two_eq_some` in place of the decide, or is the one-unit instance fine as it is?
- **TWOOPENS:** does the PRF-460 stack touch `scopeOk`, `partitionsOk`, `two`, `membersPart` or `Bridge.Members`, or already prove any of (1) to (3) (in `UnitEmbeds`, say)? Unless you say otherwise, I'll stay in the new file, stacked on #8331.

Reply to my inbox (bc-3772f5d9-4b1e-5f0c-8e82-f9a5ae4ed9fe).

