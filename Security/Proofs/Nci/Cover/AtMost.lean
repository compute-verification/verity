import Proofs.Nci.Cover.Width

/-!
# The cover theorem on blocks of at most `s` gates

`InferenceBlock s k I` asks for exactly `s` gates, while the proofs use only `#I ≤ s`, through the premises. The
hardness premises range over every set of at most `s` gates, so a premise at `s` holds at every `s' ≤ s`
(`updateHardness_mono`, `conditionalUpdateHardness_mono`), and a set of at most `s` gates is an inference block of size
its own number of gates, so each per-block bound holds for it as stated (`intensity_of_card_le`,
`additiveIntensity_of_card_le`), by the guarantee at `s' = #I`. `Width` ranges over inference blocks of exactly `s`
gates and is not monotone in `s`, so `widthIntensity_of_card_le` takes it at every size up to `s`, which the cut width
gives (`width_of_cutWidth`, with `updateHardness_mono`).
-/

namespace Nci.Cover.SecurityProofs

theorem updateHardness_mono {C : Circuit} {J L : Type} [DecidableEq L] (M : MulBlocks C J L) {s s' : ℕ} {b : ℝ}
    (h : s' ≤ s) (hH : Assumptions.UpdateHardness M s b) : Assumptions.UpdateHardness M s' b :=
  fun S hS => hH S (hS.trans h)

theorem conditionalUpdateHardness_mono {C : Circuit} {J L : Type} [DecidableEq J] [DecidableEq L]
    (M : MulBlocks C J L) (A : J → Finset (Fin C.N)) {s s' : ℕ} {b : ℝ} (h : s' ≤ s)
    (hH : Assumptions.ConditionalUpdateHardness M A s b) : Assumptions.ConditionalUpdateHardness M A s' b :=
  fun S hS => hH S (hS.trans h)

/-- **Intensity on a block of at most `s` gates**: under `UpdateHardness` at `s` and the upstream certificate with
`k < b D`, a set of at most `s` gates handed at most `k` bits besides `θ` finishes at most `k² / b + k` counted
products. -/
theorem intensity_of_card_le {C : Circuit} {J L : Type} [DecidableEq J] [DecidableEq L] (M : MulBlocks C J L)
    (A : J → Finset (Fin C.N)) {s k D : ℕ} {b : ℝ} (hb : 0 < b) (hH : Assumptions.UpdateHardness M s b)
    (hU : M.Upstream A D) (hk : (k : ℝ) < b * D) {I : Finset (Fin C.N)} (hs : I.card ≤ s)
    (hI : (C.inputs I \ C.θ).card ≤ k) : b * (M.finished I).card ≤ k * k + b * k :=
  Intensity C M A I.card k D b hb (updateHardness_mono M hs hH) hU hk I ⟨rfl, hI⟩

/-- **Intensity, additive, on a block of at most `s` gates.** -/
theorem additiveIntensity_of_card_le {C : Circuit} {J L : Type} [DecidableEq J] [DecidableEq L]
    (M : MulBlocks C J L) (A : J → Finset (Fin C.N)) {s k D : ℕ} {b : ℝ} (hb : 0 < b)
    (hCH : Assumptions.ConditionalUpdateHardness M A s b) (hU : M.Upstream A D) (hk : (k : ℝ) < b * D)
    {I : Finset (Fin C.N)} (hs : I.card ≤ s) (hI : (C.inputs I \ C.θ).card ≤ k) :
    4 * b * (M.finished I).card ≤ k * k + 4 * b * k :=
  AdditiveIntensity C M A I.card k D b hb (conditionalUpdateHardness_mono M A hs hCH) hU hk I ⟨rfl, hI⟩

/-- **Intensity, width `f`, on a block of at most `s` gates**, with `Width` at every size up to `s`. -/
theorem widthIntensity_of_card_le {C : Circuit} {J L : Type} [DecidableEq J] [DecidableEq L] (M : MulBlocks C J L)
    (A : J → Finset (Fin C.N)) {s k D f : ℕ} {b : ℝ} (hb : 0 < b)
    (hCH : Assumptions.ConditionalUpdateHardness M A s b) (hW : ∀ s' ≤ s, Assumptions.Width M A s' k f)
    (hU : M.Upstream A D) (hk : (k : ℝ) < b * D) {I : Finset (Fin C.N)} (hs : I.card ≤ s)
    (hI : (C.inputs I \ C.θ).card ≤ k) : 4 * b * f * (M.finished I).card ≤ k * k + 4 * b * f * k :=
  WidthIntensity C M A I.card k D f b hb (conditionalUpdateHardness_mono M A hs hCH) (hW I.card hs) hU hk I ⟨rfl, hI⟩

end Nci.Cover.SecurityProofs
