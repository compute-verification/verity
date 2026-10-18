import Definitions.Accounting.Confidence
import Proofs.Accounting.PouwWork

/-!
# Resource accounting's layers, as events of the actual-circuit model

Decision 111's layers stated over `Accounting.Sound`, the per-run soundness of the actual-circuit model: a strategy maps
the coins to runs, and an event is sound at `δ` when every feasible, causal strategy accepts with the event failing on at
most a `δ` fraction of the coins. An `Amount` is what a run did of one resource, in its own unit, at a set of leaves during
one window.

* `Certified` (layer 1): the developer's leaves `P` together did at least `W` of the verifier's view, at no location.
* `Capacity` (layer 2): every feasible run did at most `c` at a unit, a data-sheet premise on `𝓕`.
* `Attributed` (layer 3): each spatial unit of `U` did at least its share less its leakage.
* `residual` (layer 4): with the declared work attributed, the unit's total within its capacity, and the declared and the
  hidden work adding at the unit (each device-tick credited once), the hidden work at each unit is at most its capacity
  less its share less its leakage, except with the attribution's error. `fleet` is the same step without attribution: the
  certified work is subtracted from the capacity of the whole of `P`, which is the declared capacity only when `P` holds no
  undeclared hardware.
* `pouw_local_help` (layer 3 for compute, in PoUW's own vocabulary): `PouwLocalOff` with the node's help at most `H`
  rather than `0`, so its credits fall short of `Wb · Nmin` by at most `ρ · H`. A block's ingress bound is `H`.

No guarantee reads these yet, so they stay out of the spec.
-/

namespace Accounting.Layers

variable {L : Type*} [LinearOrder L]

/-- What a run did of one resource at a set of leaves during one window: bits read, bit-ticks held, work at a price. -/
abbrev Amount (L : Type*) := Finset L → Actual.Run L → ℕ

/-- What the verifier sees (`Actual.Run.view`). -/
abbrev View (L : Type*) := Set (ℕ × L × ℕ × ℕ × ℕ)

/-- **Certified work** (layer 1): the leaves `P` together did at least `W` of the verifier's view, sound at `δ`. -/
def Certified (Ω : Type) [Fintype Ω] (𝓕 : Set (Actual.Run L)) (V P : Finset L) (acc : View L → Prop) (μ : Amount L)
    (W : View L → ℕ) (δ : ℝ) : Prop :=
  Sound Ω 𝓕 V acc (fun S r => W ((S r).view V) ≤ μ P (S r)) δ

/-- **A capacity premise** (layer 2): every feasible run did at most `c` at `X`. -/
def Capacity (𝓕 : Set (Actual.Run L)) (μ : Amount L) (X : Finset L) (c : ℕ) : Prop := ∀ R ∈ 𝓕, μ X R ≤ c

/-- **Attribution** (layer 3): each unit `X` of `U` did at least its share less its leakage, sound at `δ`. -/
def Attributed (Ω : Type) [Fintype Ω] (𝓕 : Set (Actual.Run L)) (V : Finset L) (acc : View L → Prop) (μ : Amount L)
    (U : Finset (Finset L)) (share : View L → Finset L → ℕ) (leak : Finset L → ℕ) (δ : ℝ) : Prop :=
  Sound Ω 𝓕 V acc (fun S r => ∀ X ∈ U, share ((S r).view V) X ≤ μ X (S r) + leak X) δ

/-- An event implied, on feasible runs, by a sound one is sound at the same error. -/
theorem sound_mono {Ω : Type} [Fintype Ω] {𝓕 : Set (Actual.Run L)} {V : Finset L} {acc : View L → Prop}
    {E E' : (Ω → Actual.Run L) → Ω → Prop} {δ : ℝ} (h : Sound Ω 𝓕 V acc E δ)
    (hE : ∀ S r, S r ∈ 𝓕 → E S r → E' S r) : Sound Ω 𝓕 V acc E' δ := by
  intro S hS hc hV
  refine le_trans ?_ (h S hS hc hV)
  refine Nat.cast_le.mpr (Finset.card_le_card ?_)
  intro r hr
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hr ⊢
  exact ⟨hr.1, fun hE₀ => hr.2 (hE S r (hS r) hE₀)⟩

/-- **Joint accounting at each spatial unit** (layer 4). -/
theorem residual {Ω : Type} [Fintype Ω] {𝓕 : Set (Actual.Run L)} {V : Finset L} {acc : View L → Prop}
    {decl hid total : Amount L} {U : Finset (Finset L)} {share : View L → Finset L → ℕ} {leak c : Finset L → ℕ}
    {δ : ℝ} (hcap : ∀ X ∈ U, Capacity 𝓕 total X (c X))
    (hadd : ∀ R ∈ 𝓕, ∀ X ∈ U, decl X R + hid X R ≤ total X R)
    (hatt : Attributed Ω 𝓕 V acc decl U share leak δ) :
    Sound Ω 𝓕 V acc (fun S r => ∀ X ∈ U, hid X (S r) ≤ c X - (share ((S r).view V) X - leak X)) δ :=
  sound_mono hatt fun S r hR h X hX => by
    have h₁ := h X hX
    have h₂ := hcap X hX (S r) hR
    have h₃ := hadd (S r) hR X hX
    omega

/-- **Joint accounting without attribution**: the certified work comes off the capacity of the whole of `P`. -/
theorem fleet {Ω : Type} [Fintype Ω] {𝓕 : Set (Actual.Run L)} {V P : Finset L} {acc : View L → Prop}
    {decl hid total : Amount L} {W : View L → ℕ} {c : ℕ} {δ : ℝ} (hcap : Capacity 𝓕 total P c)
    (hadd : ∀ R ∈ 𝓕, decl P R + hid P R ≤ total P R) (hcert : Certified Ω 𝓕 V P acc decl W δ) :
    Sound Ω 𝓕 V acc (fun S r => hid P (S r) ≤ c - W ((S r).view V)) δ :=
  sound_mono hcert fun S r hR h => by
    have h₂ := hcap (S r) hR
    have h₃ := hadd (S r) hR
    omega

end Accounting.Layers

namespace Accounting.Layers

open Accounting.PouwWork

/-- **Compute attribution at a help bound** (layer 3 for PoUW): `Accounting.Guarantees.PouwLocalOff` with A5 at `H` in
place of `0`. The verifier accepts while the node's own credits outside `m`, plus `ρ · H`, fall short of `Wb · Nmin`
with probability at most `δdet + δluck + δskip`. With `H` eight times the node's ingress bytes in the window (NET-16)
and `ρ` an eighth of the work one output byte stands for, `ρ · H` is the leakage: ingress bytes times work per output
byte. -/
theorem pouw_local_help {Ω : Type} [Fintype Ω] {L : Type*} (S : Ω → Actual.Run L) (P : Verdict Ω L) (X : Finset L)
    (π : ∀ r, Fin (S r).A.N → ℕ) (m : ∀ r, Finset (Fin (S r).A.N)) (Wb Nmin ρ H : ℕ) (δdet δluck δskip : ℝ)
    (hdet : Assumptions.PouwDetection S P Nmin δdet) (hluck : Assumptions.PouwLuck S P Nmin δluck)
    (hfl : Assumptions.PouwFloorOff S P X π m Wb Nmin ρ δskip) (hH : Assumptions.PouwHelpBound S P X H) :
    Pouw.pr (fun r => Accepts S P r ∧ credits S P X π m r + ρ * H < Wb * Nmin) ≤ δdet + δluck + δskip := by
  obtain ⟨Z, hZ, hfloor⟩ := hfl
  have hsub : ∀ r, (Accepts S P r ∧ credits S P X π m r + ρ * H < Wb * Nmin) →
      ((Accepts S P r ∧ good S P r < Nmin) ∨ (Accepts S P r ∧ Nmin ≤ good S P r ∧ goodAll S P < Nmin)) ∨ Z r := by
    intro r ⟨hacc, hlt⟩
    by_cases hg : good S P r < Nmin
    · exact .inl (.inl ⟨hacc, hg⟩)
    by_cases hall : goodAll S P < Nmin
    · exact .inl (.inr ⟨hacc, Nat.le_of_not_lt hg, hall⟩)
    by_cases hz : Z r
    · exact .inr hz
    have h₁ := hfloor r hz (Nat.le_of_not_lt hall)
    have h₂ : ρ * help S P X r ≤ ρ * H := Nat.mul_le_mul le_rfl (hH r hacc)
    exact absurd (h₁.trans (Nat.add_le_add_left h₂ _)) (Nat.not_le.mpr hlt)
  calc Pouw.pr (fun r => Accepts S P r ∧ credits S P X π m r + ρ * H < Wb * Nmin)
      ≤ Pouw.pr (fun r => ((Accepts S P r ∧ good S P r < Nmin) ∨
          (Accepts S P r ∧ Nmin ≤ good S P r ∧ goodAll S P < Nmin)) ∨ Z r) := SecurityProofs.PouwWork.pr_mono hsub
    _ ≤ Pouw.pr (fun r => (Accepts S P r ∧ good S P r < Nmin) ∨
          (Accepts S P r ∧ Nmin ≤ good S P r ∧ goodAll S P < Nmin)) + Pouw.pr Z :=
        SecurityProofs.PouwWork.pr_or_le _ _
    _ ≤ (Pouw.pr (fun r => Accepts S P r ∧ good S P r < Nmin) +
          Pouw.pr (fun r => Accepts S P r ∧ Nmin ≤ good S P r ∧ goodAll S P < Nmin)) + Pouw.pr Z :=
        add_le_add_left (SecurityProofs.PouwWork.pr_or_le _ _) _
    _ ≤ δdet + δluck + δskip := add_le_add (add_le_add hdet hluck) hZ

end Accounting.Layers
