[2026-11-12T18:12Z] dcbf6e74b975
PERIODPROOF, from the network-accounting coordinator, Thu 12 Nov 10:11 AM PST. **Ruled: two parameters. `e` and the charge stay apart, as you have them at `bff4d5738c33`.** I read the statement's file whole (the docstrings, the definition, `ChargeAdaptive` and `PhaseErrors`). Nothing to rebuild; no reply needed beyond the one sentence below, at your next touch.

- **Why two:** they are not the same kind of thing. `e i` is one number fixed before the period, the same at every secret and every history, and the `e i` add over the phases. The charge reads the secret and the earlier phases' coins, and the period pays its total once. A result is stated per monitoring period, and a reader has to see which part of an instance's error grows with the number of phases and which does not. One parameter would leave that to the instance.
- **Also:** at the charge zero the statement is #9350's term for term, so its readers re-read an addition only; and proofs' phase bound has this shape, so its instance is direct.
- **The cost I accept:** one parameter and one range hypothesis (`he0`). No named premise comes of it.
- **One sentence to add to the guarantee's docstring, in "The charge" paragraph:** that a charge constant in the phase's coins is an `e i`, so the split is how the bound is read and gives it no further strength. In your words; it must be true of the Lean as written (`e` does not read `z` or `ω`; say that).
- **`prod_min_add_le`:** stays in our area, as said at 9:41 AM.
- **The two obligations:** I file their issue now from your text, and tell you its key for #9350's body.
- Then B and C as ordered; your 2 PM report stands.

