import Definitions.NetworkCertifier.Assumptions.Horizon

/-!
# The wrong units over a horizon of sampled audits

A horizon is `m` windows, each audited on its own live coins: the horizon's coins are a product `Fin m → Ω`, window `i`
drawing `k i` of its `n i` units at `ω i`. The units wrong at window `i`'s registration (`W i ω`) are fixed before its
draw: they read only the earlier windows' coins. Every wrong unit risks detection in its window, and the windows'
draws are independent, so the escapes multiply: with `q ≤ k i / n i` in every window, every window's audit accepts while
the horizon holds at least `F` wrong units in total with probability at most `(1 - q)^F` plus the windows' soundness
errors, whatever the number of windows. At `F ≥ ln (1/δ) / q` that is at most `δ + Σ_i ε_i`.
-/

namespace NetTiming.Guarantees

open Claims

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The wrong units over a horizon of accepted sampled audits.** Over `m` windows on independent live coins, window
`i` drawing a uniform `k i`-subset of its `n i` units (`uniform-draw`), with the units wrong at each registration fixed
before that window's draw, and each window's audit sound up to `ε i` (`window-sound`): for `0 ≤ q ≤ 1` with
`q · n i ≤ k i` in every window,

* every window's audit accepts and the horizon holds at least `F` wrong units with probability at most
  `(1 - q)^F + Σ_i ε i`;
* `(1 - q)^F ≤ δ` once `ln (1/δ) ≤ q · F`, so at most `ln (1/δ) / q` wrong units pass the whole horizon except with
  probability `δ + Σ_i ε i`. -/
def HorizonWrongUnitsLe : Prop :=
  ∀ {m : ℕ} {Ω : Type} [Fintype Ω] [Nonempty Ω] (n k : Fin m → ℕ) (draw : (i : Fin m) → Ω → Finset (Fin (n i)))
    (hdraw : ∀ i, DrawUniform (draw i) (k i)) (W : (i : Fin m) → (Fin m → Ω) → Finset (Fin (n i)))
    (hW : ∀ i ω ω', (∀ j, j < i → ω j = ω' j) → W i ω = W i ω')
    (acc : Fin m → (Fin m → Ω) → Prop) (ε : Fin m → ℝ)
    (hsound : ∀ i, WindowSound (acc i) (draw i) (W i) i (ε i))
    (q : ℝ) (hq0 : 0 ≤ q) (hq1 : q ≤ 1) (hq : ∀ i, q * n i ≤ k i) (F : ℕ),
    Pr (fun ω => (∀ i, acc i ω) ∧ F ≤ ∑ i, (W i ω).card) ≤ (1 - q) ^ F + ∑ i, ε i ∧
      ∀ δ : ℝ, 0 < δ → Real.log (1 / δ) ≤ q * F → (1 - q) ^ F ≤ δ

end NetTiming.Guarantees
