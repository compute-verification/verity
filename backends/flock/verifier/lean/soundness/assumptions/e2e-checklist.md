# The end-to-end theorem: the checklist of its gaps

`FlockSoundness/E2E.lean` states the end-to-end theorem as a skeleton:
- `flock_e2e_count`, the integrity profile's count curve;
- `flock_e2e_drawn`, the drawn units.

**What it says.** Suppose the audit accepts a batched session's proofs for the drawn units under the partition. Then,
except with `ε_ks(σ) + δ_link` (plus `miss_L(K)` for the count), the integrity profile holds for the committed circuit
(the program's Boolean circuit, as circuit types) at the committed values. Those are public on the constants (1) and on
the statement's zero (0), and the plurality of the prover's openings elsewhere (`Xpub`, `Audit/FlockPublic.lean`).

**What it doesn't say yet.** Two links to the executable are carried by hypotheses, not by the statement:
- "Accept" is `execAccept`, any event on the audit's outcome that implies the model's acceptance. So as pinned, the
  theorems are the model-level ones.
  - `flock-verify`'s verdict on the run's transcript is not an event on that outcome.
  - Discharging `hExec` therefore changes the statement's shape, not only the proof.
- "Each unit's rows derived by the verifier" is not in it either. `unitRows` appears in neither the conclusion nor
  `RowsL1`, and `dp.rows` says only that a unit's rows don't depend on the draw or the registration. S4's discharge of `dp`
  and `hL1` carries it.

**What it assumes, cryptographically:** A2, SHA-512 expected-time collision resistance, for the link finders built from
the prover bounded (`hCR : LinkCR … (reg σ) (cont σ) …`, `ASSUMPTIONS.md`), and nothing else. The `_hm96` forms' bounds
are at the registered leaves; read from registered roots, an audit bound adds the roots' Merkle binding (`δ_tree`, under
`cr/sha-512`, `DESIGN.md` §1).

**What it composes.** The skeleton rests on theorems already proved on `main`: `flock_batched_count_public` and
`flock_batched_drawn_public`, which carry the compiled audit, the link theorem at the public transcript
(`flock_batched_linkSoundE_pub`, from `flock_batched_linkSoundE`) and knowledge soundness.

**The gaps.** Everything not yet proved is one of the named hypotheses below. Each is discharged by the PRs listed, and
the skeleton's pins then lose that hypothesis. The pins need a named statement reviewer (red team, bc-f0bc7e75) each
time.

| Hypothesis | What it says | Owner | Discharged by |
|---|---|---|---|
| `hExec` | the audit's acceptance event implies the model's: `∀ o, execAccept o → o.1 = true` | the refinement lane, "Prove verifier refines Lean model" (bc-159ce83b) | its refinement theorem, game-level (R11), which restates the pins with the executable's verdict on the run's transcript |
| `dp : DerivedPlaces` | each drawn unit's place in its table (`UnitPlace`), with its rows the verifier's: `(place S R u h).R = unitRows u`. Its input columns on one gate carry one value in every satisfying witness (`UnitPlace.aliased`) | audit-lean (1d); flock-verifier (1e); flock-soundness (S4) | 1d's `placement_compose` from `setupH`; 1e, the verifier folding `deriveChecked`'s rows; S4, `unitRows u = Rows.compose (deriveChecked …)` for the partition's units. `aliased` from each slot input's copy position and the forced-zero rows as the verifier's Δ writes them, both kinds (the unit's own and the range's first slot's): inputs on one source copy one position, and the zero's inputs copy forced-zero rows. It is sound because Δ has one source per destination, which the verifier's exactly-once checks keep |
| `hL1 : RowsL1 … ones` | at values carrying 1 on `ones` (the gates the units' constants read), a unit wrong on the program's Boolean circuit is wrong on its rows: `PB.wrong (proj X) ⊆ P.wrong X` | flock-soundness | `compose_sound` (#205, flat types) and `Types.Dag.unit_sound` (#247, type DAGs), both at the constant 1; `flatten_eval` and W6 (S4) |
| `hConst : ConstCols … ones` | at each drawn unit's place, only the constant's column sits on a gate of `ones`: no unit's input column reads a constant. With it every satisfying message decodes `ones` to 1 (`UnitPlace.decode_one`, `pubDecodes_of_place`), so the committed transcript is 1 there | flock-soundness (S4) | for a program of units placed at its tables' classes, `UProg.constCols_of_classes` (proved: each unit's constant column is wired to `one`, `pairs_oneGate`, and an instance shares a gate only between input columns, `constCols_singleton`); for another partition, its places |
| `hZero : ZeroCols … zeros` | at each drawn unit's place, every column on a gate of `zeros` (the statement's zero) carries 0 in every satisfying witness, so the committed transcript is 0 there | audit-lean (1d); flock-verifier (1e); flock-soundness (S4) | the forced-zero rows as the verifier's Δ writes them: each slot input that reads the zero copies a row of `TableClass.zeros`, which `TableClass.zero` holds at 0. For a program placed at its tables' classes, `UProg.zeroCols_of_classes` proves it for the program input `zer` (#316's zero) from one more fact of the class data, `hzr`: each slot input that reads `zer` copies a forced-zero row, which 1d and 1e give from Δ. `UProg.flock_e2e_count_classes_zero` and `_drawn_classes_zero` apply it; the `_classes` forms leave the zero at the plurality (`zeros = ∅`) |
| `vb : ValueBinding` | the wires bind to their registered commit strings (already named on `main`) | lean-value-binding; M0's `hm96-sha512` leaf layout | built from the serving rows by `Binding.HmRows.binding` (`FlockSoundness/Binding/`), with `collide_spec` proved (`ExplicitLeaf.hm96`: an explicit SHA-512 collision). `flock_e2e_count_hm96`, `flock_e2e_drawn_hm96` and their `_exec_hm96` forms take, in its place, `HmRowComputes` and the layout's facts (the next row) |
| `hHm : HmRowComputes` (in `_hm96`) | the circuit's row hashing computes `hm96-sha512/row/v1`: in a satisfying witness, the `b ‖ c` a row's `hm96` slot outputs is the commit string of the row and salt its `sha512x3` and `hm96` slots read (`Assumptions.HmRowComputes`, not cryptographic) | the verified-lowering lane (phase 2i) | only with readers of the witness at fixed addresses (`rowAt`, `saltAt`, `bcAt`), a proof from the `sha512x3` and `hm96` sections of the circuit file: each slot's constraints force SHA-512's compression (and hm96's mask and salt digest), and the chain wiring (`HmRow.rowWires`) from the row prefix's midstate gives SHA-512 of the row |
| `lay : Layout`, `rs : HmRows` (in `_hm96`) | the layout's facts: the decoder reads the rows the witness holds (`decode_row`); the `b ‖ c` the witness holds at a row hashes to the registered leaf (`bc_registered`: the statement's regions pin the `hm96` outputs to the public `b ‖ c`, which `HmRow.checkPublic` hashes into the registered frame-v3-sha512 roots); the returned outputs' leaves (`other_registered`) | lean-value-binding, with flock-verifier (S4) | from the placement (`DerivedPlaces`, `UnitPlace`) and the refinement of `HmRow.regions` and `checkPublic` (R11). Discharging `bc_registered` and `other_registered` from registered roots also pays the roots' Merkle binding: `δ_tree`, under `cr/sha-512` (`DESIGN.md` §1), which the `_hm96` bound leaves out. `decode_row` counts as discharged only with the fixed readers `rowAt`, `saltAt` |
| `hCR` (A2) | SHA-512 expected-time collision resistance for the link finders of the prover bounded, `σ` (`LinkCR … (reg σ) (cont σ) …`), not of every prover at once | the assumption of record | not discharged: it is the theorem's cryptographic assumption |

**Why the constant is public.** The rows multiply outputs by the constant: each output is a copy row `(form) · 1`, and
the constant's row is `1 · 1`. At a transcript whose constant is 0, a unit's rows hold with every output 0 while the
Boolean circuit's outputs are the forms. So `RowsL1` can hold only where the constant is 1 (red-team review of #207).
- **The committed transcript is 1 on `ones`.** The plurality of the prover's openings (`Xplur`) can't carry that on its
  own: for a strategy whose trials never read the constant's commit string, `Xval` is an arbitrary argmax. But every
  satisfying message of a drawn unit's table decodes the constant to 1, so the link event never fires on it
  (`linkEvB_of_pub`), and the link bound holds unchanged at `Xpub` for every strategy.
- **One `ones` for both.** `hL1` and `hConst` constrain the same parameter from opposite sides: a set too small makes
  `hL1` false, and a set too large makes `hConst` false (an input column would sit on a gate of it). S4 discharges both
  for exactly the gates the units' constants read.

**Why the zero is different.** The zero multiplies no row, so `RowsL1` holds at any value of it (#316). But "wrong at
the committed values" judges a unit that reads the zero at the zero's committed value: at a committed zero of 1, a padded
attention unit would be judged against `b = 1` (red-team review of #316, C1). At `Xpub` the committed zero is 0, which
`hZero` makes sound: every satisfying message decodes it to 0. So a claim may read the profile as being about the zero
padding when `hZero` is discharged, and not before.

**How to use this list.**
- Each PR that discharges a hypothesis changes the skeleton to prove it: an argument becomes a theorem applied. It also
  deletes that row here. When only `hCR` remains, the end-to-end theorem is proved.
- The numeric side conditions (`hk`, `hRw`, `hM`, `hρ`, `hr`, `ht`) are parameters of the bound, not gaps. They are the
  same as in `flock_batched_count_placed`. `ones` and `zeros`, the statement's public gates, are parameters too.
- `ε_ks(σ)` is the knowledge-soundness term (`ksAvgBE`), which `flock_batched_knowledgeSoundE` bounds. `δ_link` is
  `linkBoundE`, the link theorem's `Q_s/(1 − ρ)·(2t'(1 + k)/2^256 + 1/(eM) + k/(eR_w))`.
