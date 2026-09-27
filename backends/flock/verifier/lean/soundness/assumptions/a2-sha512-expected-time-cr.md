# A2. Expected-time collision resistance of SHA-512

In CDGSY24's expected-time sense (ePrint 2024/1434), for the audit's value layer (Daniel, Sep 27; constant corrected
the same day). `FlockSoundness/Assumptions.lean` states it as `Assumptions.SHA512ExpectedTimeCR`, and the link theorem
(`flock_batched_linkSoundE`) and the composed audit (`FlockSoundness/Audit/FlockLinked.lean`) take it as `hCR`, once for
each commit string's explicit finder.

- **Statement.** A collision finder whose expected number of SHA-512 evaluations is `T` succeeds with probability at
  most `T/2^256`. `SHA512ExpectedTimeCR` states it for one finder: a game with a fixed strategy, whose outcomes
  record the pair output and the evaluations made, `t` for each run of the prover.
- **Why it is plausible.** A random oracle meets it with 0.33 bits to spare: its sharp bound is `T/E[τ_N]`, and
  hashing until the first collision costs `E[τ_N] ≈ √(π/2)·2^256 = 2^256.33` (`../DESIGN.md` §3). The earlier `T/2^256.5`
  came from the fixed-budget bound `min(1, q²/2^513) ≤ q/2^256.5`, which that attack beats. It is linear in `T`
  because a finder can gamble.
- **Per finder.** Some finder outputs a collision after no evaluations, since collisions exist. So the `Prop` is not
  stated for all finders: the link theorem takes it for the finder it builds from the prover.
- **Where it is used.** Only the link term, `t·Q_s(8N₀/e)/2^256` per audit, with `Q_s ≤ N_s` the commit strings
  the drawn units read (audit B `t·2^-209.4`, `2^-129.4` at `t = 2^80`). The strict-time `2√(t·2^-221.5)` and the
  random oracle's `t²/2^511` are comparisons (`../DESIGN.md` §3).
  The compiled theorem, `δ_tree` and the knowledge term's conflicts keep strict-time `cr/sha-512` (`cryptography-and-coins.md`).

