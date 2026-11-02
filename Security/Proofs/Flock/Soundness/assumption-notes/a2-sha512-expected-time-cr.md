# A2: why it is plausible, and where it is used

The statement is in `../ASSUMPTIONS.md`.

- **Why it is plausible.** A random oracle meets it with 0.33 bits to spare: its sharp bound is `T/E[τ_N]`, and
  hashing until the first collision costs `E[τ_N] ≈ √(π/2)·2^256 = 2^256.33` (`../DESIGN.md` §3). The earlier `T/2^256.5`
  came from the fixed-budget bound `min(1, q²/2^513) ≤ q/2^256.5`, which that attack beats. It is linear in `T`
  because a finder can gamble.
- **Where it is used.** Only the link term, `t·Q_s(8N₀/e)/2^256` per audit, with `Q_s ≤ N_s` the commit strings
  the sampled proof blocks read (audit B `t·2^-209.4`, `2^-129.4` at `t = 2^80`). Under strict-time `cr/sha-512` alone it is
  proved for the non-ZK session (not `--zk`'s, `../ASSUMPTIONS.md`) at `t^{2/3}·2^-123.5` (`2^-70.2` at
  `t = 2^80`; `../CROnly/`), and the random oracle's `t²/2^511` is a comparison (`../DESIGN.md` §3).
  The compiled theorem, `δ_tree` and the knowledge term's conflicts keep strict-time `cr/sha-512` (`../ASSUMPTIONS.md`).
