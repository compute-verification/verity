# The end-to-end theorem: the checklist of its gaps

`FlockSoundness/E2E.lean` states the end-to-end theorem as a skeleton:
- `flock_e2e_count`, the integrity profile's count curve;
- `flock_e2e_drawn`, the drawn units.

**What it says.** Suppose the executable verifier accepts a batched session's proofs for the drawn units under the
partition, with each unit's rows derived by the verifier (`Flock.Derive.derive`). Then, except with
`ε_ks(σ) + δ_link` (plus `miss_L(K)` for the count), the integrity profile holds for the committed circuit (the
program's Boolean circuit, as circuit types) at the committed values.

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
| `hExec` | the executable verifier's acceptance implies the model's: `∀ o, execAccept o → o.1 = true` | the refinement lane, "Prove verifier refines Lean model" (bc-159ce83b) | its refinement theorem, with `execAccept` defined from `flock-verify`'s verdict on the run's transcript |
| `dp : DerivedPlaces` | each drawn unit's place in its table (`UnitPlace`), with its rows the verifier's: `(place S R u h).R = unitRows u` | audit-lean (1d); flock-verifier (1e); flock-soundness (S4) | 1d's `placement_compose` from `setupH`; 1e, the verifier folding `derive`'s rows; S4, `unitRows u = Rows.compose (derive …)` for the partition's units |
| `hL1 : RowsL1` | a unit wrong on the program's Boolean circuit is wrong on its rows: `PB.wrong (proj X) ⊆ P.wrong X` | flock-soundness | `compose_sound` (#205, flat types; S3, type DAGs); `flatten_eval` and W6 (S4) |
| `vb : ValueBinding` | the wires bind to their registered commit strings (already named on `main`) | M0's `hm96-sha512` leaf layout | the row hashing's circuit side (verified-lowering design, phase 2, 2i) |
| `hCR` (A2) | SHA-512 expected-time collision resistance for the link finders | the assumption of record | not discharged: it is the theorem's cryptographic assumption |

**How to use this list.**
- Each PR that discharges a hypothesis changes the skeleton to prove it: an argument becomes a theorem applied. It also
  deletes that row here. When only `hCR` remains, the end-to-end theorem is proved.
- The numeric side conditions (`hk`, `hRw`, `hM`, `hρ`, `hr`, `ht`) are parameters of the bound, not gaps. They are the
  same as in `flock_batched_count_placed`.
- `ε_ks(σ)` is the knowledge-soundness term (`ksAvgBE`), which `flock_batched_knowledgeSoundE` bounds. `δ_link` is
  `linkBoundE`, the link theorem's `Q_s/(1 − ρ)·(2t'(1 + k)/2^256 + 1/(eM) + k/(eR_w))`.
