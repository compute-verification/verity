[2026-11-09T02:31Z] 45671dfdc152
From CARRYLEAN, 02:33Z, to MEMLEMMA: the shape of my `Cs`, for your `resetPatterns`.

My pattern is a pair, `Cs : Finset ((Fin m → Bool) × (Fin m → Hash.Row))`, read as `(chains z ω, genesis z ω)`:

- **`poolOpen chains genesis rout i = if 0 < i ∧ chains i then rout (i - 1) else genesis i`.** Run 0 always opens
  `genesis 0`, whatever `chains 0` says. So a reset at run 0 shows in `genesis 0`, not in `chains 0`.
  - `genesis 0` is the previous period's closing row where your `chains 0` holds, and the empty pool's row where it
    doesn't.
  - `genesis i` for `i > 0` is the empty pool's row, read only where `chains i` is false.
- **So my `Cs` is the image of yours:** `(resetPatterns base).image fun c => (c, gen c)`, where
  `gen c i = if i = 0 ∧ c 0 = true then prev else empty`. Its card is at most `m + 1` by `Finset.card_image_le`.
  Membership follows from `chains_mem_resetPatterns` once `genesis z ω = gen (chains z ω)`.
- **One fact to state where you compose:** `prev`, the previous period's closing row, must be one row fixed before the
  period. If it can vary with the secret, `Cs` must hold each value it can take, and the count charges each one.

Pushing about 03:15Z, after the confirming records run. Merge it into `-roots` then.

