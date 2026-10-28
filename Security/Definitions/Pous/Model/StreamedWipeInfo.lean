import Definitions.Pous.Model.StreamedWipe

/-!
# The streamed-label wipe: what an accepted run holds, and guessing lists (experimental)

The vocabulary of `Pous.Guarantees.StreamedWipeMeetsInfo`, which states the wipe's retention bound as a conditional
mutual information.

- `held` is what the server holds when the fill ends, as the analysis of a run sees it: the fill's `(σ, ξ)` when the
  challenger accepts, and `none` (⊥) when it rejects.
- `listAdvantage` is the PRF's advantage against every distinguisher that guesses the prior content `z` as a list of
  one fixed size, read from `held` and the challenges.
-/

namespace Pous.StreamedWipe

open scoped ENNReal

open Classical in
/-- What the server holds when the fill ends, as a run's analysis sees it: `fill z L`, the `M` bits inside the boundary
and the `E` bits of egress, when the challenger accepts `P` on `(z, L, c)`, and `none` (⊥) when it rejects. -/
noncomputable def held {N k b q M E : ℕ} (P : Prover N k b q M E) (z : Bits q) (L : Fin N → Bits b)
    (c : Fin k → Fin N) : Option (Bits M × Bits E) :=
  if Accepts P z L c then some (P.fill z L) else none

/-- The distinguisher that a list `ℓ` of guesses of the prior content makes of `P`: with coins `(z, c)` and labels `L`,
it outputs whether `z` is among the guesses `ℓ` reads from what the server holds (`held`) and the challenges. -/
noncomputable def listDistinguisher {N k b q M E : ℕ} (P : Prover N k b q M E)
    (ℓ : Option (Bits M × Bits E) → (Fin k → Fin N) → Finset (Bits q)) :
    (Bits q × (Fin k → Fin N)) × (Fin N → Bits b) → Prop :=
  fun x => x.1.1 ∈ ℓ (held P x.1.1 x.2 x.1.2) x.1.2

/-- The PRF's advantage against guessing lists: the largest `prfAdvantage prf (listDistinguisher P ℓ)` over every `m`
and every list `ℓ` that names exactly `m` guesses whatever it reads.

A list of one guess read from an accepted run is the test `wipeDistinguisher` makes with `P.keep`, so this is at least
that advantage for every `keep` (up to the rejected runs' `2^-q`). A prover whose fill ignores `z` scores 0: then `z`
is independent of `(held, c)` in both games, and every such list holds `z` with probability `m / 2^q` in each. -/
noncomputable def listAdvantage {N k b q M E κ : ℕ} (prf : Bits κ → Fin N → Bits b) (P : Prover N k b q M E) :
    ℝ≥0∞ :=
  ⨆ (m : ℕ) (ℓ : Option (Bits M × Bits E) → (Fin k → Fin N) → Finset (Bits q)) (_ : ∀ s c, (ℓ s c).card = m),
    prfAdvantage prf (listDistinguisher P ℓ)

end Pous.StreamedWipe
