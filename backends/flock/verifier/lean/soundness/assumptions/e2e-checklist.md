# The end-to-end theorem: the checklist of its gaps

`FlockSoundness/E2E.lean` states the end-to-end theorem as a skeleton:
- `flock_e2e_count`, the integrity profile's count curve;
- `flock_e2e_drawn`, the drawn units.

**What it says.** Suppose the audit accepts a batched session's proofs for the drawn units under the partition. Then,
except with `ε_ks(σ) + δ_link` (plus `miss_L(K)` for the count), the integrity profile holds for the committed circuit
(the program's Boolean circuit, as circuit types) at the committed values.

**What it doesn't say yet.** Two links to the executable are carried by hypotheses, not by the statement:
- "Accept" is `execAccept`, any event on the audit's outcome that implies the model's acceptance. So as pinned, the
  theorems are the model-level ones.
  - `flock-verify`'s verdict on the run's transcript is not an event on that outcome.
  - Discharging `hExec` therefore changes the statement's shape, not only the proof.
- "Each unit's rows derived by the verifier" is not in it either. `unitRows` appears in neither the conclusion nor
  `RowsL1`, and `dp.rows` says only that a unit's rows don't depend on the draw or the registration. S4's discharge of `dp`
  and `hL1` carries it.

**What it assumes, cryptographically:** A2, SHA-512 expected-time collision resistance (`hCR`, `ASSUMPTIONS.md`), and
nothing else.

**What it composes.** The skeleton rests on theorems already proved on `main`: `flock_batched_count_placed` and
`flock_batched_drawn_placed`, which carry the compiled audit, the link theorem (`flock_batched_linkSoundE`) and
knowledge soundness.

**The gaps.** Everything not yet proved is one of the named hypotheses below. Each is discharged by the PRs listed, and
the skeleton's pins then lose that hypothesis. The pins need a named statement reviewer (red team, bc-f0bc7e75) each
time.

| Hypothesis | What it says | Owner | Discharged by |
|---|---|---|---|
| `hExec` | the audit's acceptance event implies the model's: `∀ o, execAccept o → o.1 = true` | the refinement lane, "Prove verifier refines Lean model" (bc-159ce83b) | its refinement theorem, game-level (R11), which restates the pins with the executable's verdict on the run's transcript |
| `dp : DerivedPlaces` | each drawn unit's place in its table (`UnitPlace`), with its rows the verifier's: `(place S R u h).R = unitRows u`. Its input columns on one gate carry one value in every satisfying witness (`UnitPlace.aliased`) | audit-lean (1d); flock-verifier (1e); flock-soundness (S4) | 1d's `placement_compose` from `setupH`; 1e, the verifier folding `deriveChecked`'s rows; S4, `unitRows u = Rows.compose (deriveChecked …)` for the partition's units. `aliased` from each slot input's copy position and the forced-zero rows as the verifier's Δ writes them, both kinds (the unit's own and the range's first slot's): inputs on one source copy one position, and the zero's inputs copy forced-zero rows. It is sound because Δ has one source per destination, which the verifier's exactly-once checks keep |
| `hL1 : RowsL1 … ones` | at values carrying 1 on `ones` (the gates the units' constants read), a unit wrong on the program's Boolean circuit is wrong on its rows: `PB.wrong (proj X) ⊆ P.wrong X` | flock-soundness | `compose_sound` (#205, flat types) and `Types.Dag.unit_sound` (#247, type DAGs), both at the constant 1; `flatten_eval` and W6 (S4) |
| `hOne` | the committed values carry 1 on `ones`: `∀ g ∈ ones, Xplur … g = true` | flock-soundness, with audit-lean | every opening of the constant's commit string carries 1 on it (`UnitPlace.decode_one` through `ValueBinding.opening_spec`), so the plurality does whenever the string is read. The case where it is never read needs care, since `Xval` then falls back to an arbitrary argmax. Alternatively the constant becomes a public wire of the value binding |
| the zero's binding (a condition on claims; not yet a hypothesis) | the committed values carry 0 on the statement's zero, the program input its forced-zero rows carry: `Xplur … zer = false` | flock-soundness, with audit-lean; the refinement lane if R11 adds `hZero` | `hOne`'s argument: every opening of the zero's positions carries 0 (the forced-zero rows, through `ValueBinding.opening_spec`), so the plurality does whenever the zero is read; the never-read case needs the same care. #207 takes it as `hZero` when R11 restates it, or the zero becomes a public wire of the value binding. Until then, no claim may read the profile as being about the zero padding |
| `vb : ValueBinding` | the wires bind to their registered commit strings (already named on `main`) | M0's `hm96-sha512` leaf layout | the row hashing's circuit side (verified-lowering design, phase 2, 2i) |
| `hCR` (A2) | SHA-512 expected-time collision resistance for the link finders | the assumption of record | not discharged: it is the theorem's cryptographic assumption |

**Why `hOne` is separate.** The rows multiply outputs by the constant: each output is a copy row `(form) · 1`, and the
constant's row is `1 · 1`. At a transcript whose constant is 0, a unit's rows hold with every output 0 while the Boolean
circuit's outputs are the forms. So `RowsL1` can hold only where the constant is 1, and the committed transcript's
constant is a gap of its own (red-team review of #207).
- **One `ones` for both.** `hL1` and `hOne` constrain the same parameter from opposite sides: a set too small makes `hL1`
  false, and a set too large makes `hOne` false. S4 discharges both for exactly the gates the units' constants read.
- **`hOne` holds per strategy.** The final theorem needs it for every `σ`. The never-read case is where that needs work.
  Making the constant a public wire of the value binding would avoid it.

**Why the zero is different.** The zero multiplies no row, so `RowsL1` holds at any value of it, and #207's bound needs
no `hZero` (#316). But "wrong at the committed values" judges a unit that reads the zero at the zero's committed value:
at a committed zero of 1, a padded attention unit would be judged against `b = 1`. So the zero's binding is a condition
on claims. None may read the profile as being about the zero padding until it's bound (red-team review of #316, C1).

**How to use this list.**
- Each PR that discharges a hypothesis changes the skeleton to prove it: an argument becomes a theorem applied. It also
  deletes that row here. When only `hCR` remains, the end-to-end theorem is proved.
- The numeric side conditions (`hk`, `hRw`, `hM`, `hρ`, `hr`, `ht`) are parameters of the bound, not gaps. They are the
  same as in `flock_batched_count_placed`.
- `ε_ks(σ)` is the knowledge-soundness term (`ksAvgBE`), which `flock_batched_knowledgeSoundE` bounds. `δ_link` is
  `linkBoundE`, the link theorem's `Q_s/(1 − ρ)·(2t'(1 + k)/2^256 + 1/(eM) + k/(eR_w))`.
