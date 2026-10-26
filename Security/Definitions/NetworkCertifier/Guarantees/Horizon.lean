import Definitions.NetworkCertifier.SampledAudit

/-!
# The wrong units over a horizon of sampled audits

A horizon is `m` windows, each audited on its own live coins: the horizon's coins are a product `Fin m → Ω`, window `i`
drawing `k i` of its `n i` units at `ω i`, a uniform `k i`-subset. The units wrong at window `i`'s registration
(`W i ω`) are fixed before its draw: they read only the earlier windows' coins. Every wrong unit risks detection in its
window, and the windows' draws are independent, so the escapes multiply: with `q ≤ k i / n i` in every window, every
window's audit accepts while the horizon holds at least `F` wrong units in total with probability at most `(1 - q)^F`
plus the windows' soundness errors `ε i`, whatever the number of windows. At `F ≥ ln (1/δ) / q` that is at most
`δ + Σ_i ε_i`.

The statement is a fact of probability over every draw, acceptance event and error: the draw's law and each window's
soundness are its hypotheses. A deployment's instances of them (A3 for the draw, `Claims.Sampling.KS` for each
window) come in where it is applied to the sampled verifier.
-/

namespace NetTiming.Guarantees

open Claims

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The wrong units over a horizon of accepted sampled audits.** Over `m` windows on independent live coins, window
`i` drawing a uniform `k i`-subset of its `n i` units (every coin draws `k i` units, and each `k i`-subset is drawn by
the share `1 / C(n i, k i)` of the coins), with the units wrong at each registration fixed before that window's draw,
and each window's audit accepting while its draw meets a wrong unit with probability at most `ε i`: for `0 ≤ q ≤ 1`
with `q · n i ≤ k i` in every window,

* every window's audit accepts and the horizon holds at least `F` wrong units with probability at most
  `(1 - q)^F + Σ_i ε i`;
* `(1 - q)^F ≤ δ` once `ln (1/δ) ≤ q · F`, so at most `ln (1/δ) / q` wrong units pass the whole horizon except with
  probability `δ + Σ_i ε i`. -/
def HorizonWrongUnitsLe : Prop :=
  ∀ {m : ℕ} {Ω : Type} [Fintype Ω] [Nonempty Ω] (n k : Fin m → ℕ) (draw : (i : Fin m) → Ω → Finset (Fin (n i)))
    (hcard : ∀ i x, (draw i x).card = k i)
    (hunif : ∀ i (S : Finset (Fin (n i))), S.card = k i →
      (Finset.univ.filter fun x => draw i x = S).card * (n i).choose (k i) = Fintype.card Ω)
    (W : (i : Fin m) → (Fin m → Ω) → Finset (Fin (n i)))
    (hW : ∀ i ω ω', (∀ j, j < i → ω j = ω' j) → W i ω = W i ω')
    (acc : Fin m → (Fin m → Ω) → Prop) (ε : Fin m → ℝ)
    (hsound : ∀ i, Pr (fun ω => acc i ω ∧ ¬Disjoint (draw i (ω i)) (W i ω)) ≤ ε i)
    (q : ℝ) (hq0 : 0 ≤ q) (hq1 : q ≤ 1) (hq : ∀ i, q * n i ≤ k i) (F : ℕ),
    Pr (fun ω => (∀ i, acc i ω) ∧ F ≤ ∑ i, (W i ω).card) ≤ (1 - q) ^ F + ∑ i, ε i ∧
      ∀ δ : ℝ, 0 < δ → Real.log (1 / δ) ≤ q * F → (1 - q) ^ F ≤ δ

end NetTiming.Guarantees
