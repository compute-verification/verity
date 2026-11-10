module

public import Definitions.Nci.Guarantees.CompliesCK
public import Proofs.Core.Expand
public import Mathlib.Tactic.Linarith

/-!
# Proofs of NCI's policy as a `(c, k)`-partition

A Boolean declaration's novel reads are its unregistered input gates, so a complying IU of `|I| ≤ Γ` gates reads at
most `α |I| ≤ α Γ` of them, and the division is a `(Γ, ⌊αΓ⌋)`-partition. Through the bit-level expansion,
an IU's size becomes its width, at least `|I|` and at most `W |I|`, and its novel reads do not grow (`ExpandSize`).

With its constants registered (`RegisteredCircuit.constInputs`) a declaration needs neither assumption:
`constInputs_isCK` and `constInputs_expand_isCK` take a complying division to those partitions with only the gates'
widths bounded, since a constant leaves its IU as a registered input and every other gate reads what it read.
-/

@[expose] public section

namespace Verity

open Finset

theorem Circuit.width_eq_card {C : Circuit} (hC : C.Boolean) (S : Finset (Fin C.N)) : C.width S = S.card := by
  simp [Circuit.width, hC.one_bit]

theorem Circuit.card_le_width (C : Circuit) (S : Finset (Fin C.N)) (h : ∀ g ∈ S, 1 ≤ C.w g) :
    S.card ≤ C.width S := by
  rw [card_eq_sum_ones]
  exact sum_le_sum h

theorem Circuit.width_le (C : Circuit) (S : Finset (Fin C.N)) (W : ℕ) (h : ∀ g, C.w g ≤ W) :
    C.width S ≤ S.card * W := by
  calc ∑ g ∈ S, C.w g ≤ ∑ _g ∈ S, W := sum_le_sum fun g _ => h g
    _ = S.card * W := by rw [sum_const, smul_eq_mul]

end Verity

namespace Nci.SecurityProofs

open Verity Finset

theorem CompliesCK : Guarantees.CompliesCK := by
  intro Γ α d hB ⟨B, P, hP⟩
  have hk : ∀ u, (d.unregistered (P.gates u)).card ≤ ⌊α * Γ⌋₊ := by
    intro u
    obtain ⟨hs, hr⟩ := hP u
    have hr' : ((d.unregistered (P.gates u)).card : ℝ) ≤ α * (P.gates u).card := by
      rw [← Circuit.width_eq_card hB]
      exact hr
    rcases Nat.eq_zero_or_pos (d.unregistered (P.gates u)).card with h0 | hpos
    · rw [h0]
      exact Nat.zero_le _
    · apply Nat.le_floor
      have hs' : ((P.gates u).card : ℝ) ≤ Γ := by exact_mod_cast hs
      have hpos' : (0 : ℝ) < (d.unregistered (P.gates u)).card := by exact_mod_cast hpos
      have hα : 0 < α := by
        by_contra hneg
        have : α * (P.gates u).card ≤ 0 :=
          mul_nonpos_of_nonpos_of_nonneg (not_lt.1 hneg) (Nat.cast_nonneg _)
        linarith
      calc ((d.unregistered (P.gates u)).card : ℝ) ≤ α * (P.gates u).card := hr'
        _ ≤ α * Γ := mul_le_mul_of_nonneg_left hs' hα.le
  exact ⟨B, P, ⟨hB, fun u => (hP u).1, hk⟩⟩

theorem ExpandComplies : Guarantees.ExpandComplies := by
  intro Γ W α d hα hw hW ⟨B, P, hP⟩
  refine ⟨B, P.expand, fun u => ?_⟩
  obtain ⟨hsize, hunreg⟩ := (Verity.SecurityProofs.ExpandSize d B P).2 u
  obtain ⟨hs, hr⟩ := hP u
  have hcard : (P.gates u).card ≤ d.D.width (P.gates u) :=
    Circuit.card_le_width _ _ fun g hg => hw g (P.isInput_of_mem_gates hg)
  have hsize' : d.expand.D.size (P.expand.gates u) = d.D.width (P.gates u) := hsize
  have hnovel : novelReads d.expand (P.expand.gates u) ≤ novelReads d (P.gates u) := by
    have : novelReads d.expand (P.expand.gates u) = (d.expand.unregistered (P.expand.gates u)).card := by
      simp [novelReads, Circuit.width, RegisteredCircuit.unregistered]
    rw [this]
    exact hunreg
  refine ⟨?_, ?_⟩
  · rw [hsize']
    calc d.D.width (P.gates u) ≤ (P.gates u).card * W := Circuit.width_le _ _ W hW
      _ ≤ Γ * W := Nat.mul_le_mul_right _ hs
  · rw [hsize']
    calc (novelReads d.expand (P.expand.gates u) : ℝ) ≤ novelReads d (P.gates u) := by exact_mod_cast hnovel
      _ ≤ α * (P.gates u).card := hr
      _ ≤ α * d.D.width (P.gates u) := mul_le_mul_of_nonneg_left (by exact_mod_cast hcard) hα

theorem floor_bound {x r K : ℕ} {α : ℝ} (hx : (x : ℝ) ≤ α * r) (hr : 0 < x → r ≤ K) : x ≤ ⌊α * K⌋₊ := by
  rcases Nat.eq_zero_or_pos x with h0 | hpos
  · rw [h0]
    exact Nat.zero_le _
  · apply Nat.le_floor
    have hpos' : (0 : ℝ) < x := by exact_mod_cast hpos
    have hα : 0 < α := by
      by_contra hneg
      have : α * r ≤ 0 := mul_nonpos_of_nonpos_of_nonneg (not_lt.1 hneg) (Nat.cast_nonneg _)
      linarith
    calc (x : ℝ) ≤ α * r := hx
      _ ≤ α * K := mul_le_mul_of_nonneg_left (by exact_mod_cast hr hpos) hα.le

/-- A unit of `P` with `d`'s constants registered has at most the gates of its unit in `P`, and unregistered reads
at most as wide as that unit's novel reads. -/
theorem constInputs_bounds (d : RegisteredCircuit) {n : ℕ} (P : Partition d.D n) (u : Fin n) :
    (P.constInputs.gates u).card ≤ (P.gates u).card ∧
      d.constInputs.D.width (d.constInputs.unregistered (P.constInputs.gates u)) ≤ novelReads d (P.gates u) := by
  refine ⟨card_le_card fun g hg => (P.mem_constInputs_gates.1 hg).2, ?_⟩
  show ∑ g ∈ d.constInputs.unregistered (P.constInputs.gates u), d.D.w g ≤
    ∑ g ∈ d.D.into (P.gates u) \ d.θ, d.D.w g
  exact sum_le_sum_of_subset (d.constInputs_unregistered P u)

/-- **A complying division is a `(Γ, ⌊αΓ⌋)`-partition with the constants registered**, on a declaration whose gates
are one bit wide: registering the constants makes it Boolean (`Circuit.constInputs_boolean`), takes them out of
their units and adds no unregistered read. -/
theorem constInputs_isCK {Γ : ℕ} {α : ℝ} {d : RegisteredCircuit} {n : ℕ} {P : Partition d.D n}
    (hP : ∀ u, d.D.size (P.gates u) ≤ Γ ∧ (novelReads d (P.gates u) : ℝ) ≤ α * d.D.size (P.gates u))
    (h1 : ∀ g, d.D.w g = 1) : P.constInputs.IsCK d.constInputs Γ ⌊α * Γ⌋₊ := by
  have hB : d.constInputs.D.Boolean := d.D.constInputs_boolean h1
  refine ⟨hB, fun u => (constInputs_bounds d P u).1.trans (hP u).1, fun u => ?_⟩
  have hw := (constInputs_bounds d P u).2
  rw [Circuit.width_eq_card hB] at hw
  have hw' : ((d.constInputs.unregistered (P.constInputs.gates u)).card : ℝ) ≤ novelReads d (P.gates u) := by
    exact_mod_cast hw
  exact floor_bound (hw'.trans (hP u).2) fun _ => (hP u).1

/-- **A complying division expands to a `(ΓW, ⌊αΓW⌋)`-partition with the constants registered**, on a declaration
whose gates are at most `W` bits wide: registering the constants gives positive fan-in in bits
(`Circuit.constInputs_bitFaninPos`), so the expansion is Boolean, and an expanded unit has at most `W` bits a gate
and reads no more than its unit's novel reads (`ExpandSize`). -/
theorem constInputs_expand_isCK {Γ W : ℕ} {α : ℝ} {d : RegisteredCircuit} {n : ℕ} {P : Partition d.D n}
    (hP : ∀ u, d.D.size (P.gates u) ≤ Γ ∧ (novelReads d (P.gates u) : ℝ) ≤ α * d.D.size (P.gates u))
    (hW : ∀ g, d.D.w g ≤ W) :
    P.constInputs.expand.IsCK d.constInputs.expand (Γ * W) ⌊α * (Γ * W : ℕ)⌋₊ := by
  refine ⟨d.D.constInputs.expand_boolean d.D.constInputs_bitFaninPos, fun u => ?_, fun u => ?_⟩
  · obtain ⟨hsize, -⟩ := (Verity.SecurityProofs.ExpandSize d.constInputs n P.constInputs).2 u
    calc (P.constInputs.expand.gates u).card = d.D.width (P.constInputs.gates u) := hsize
      _ ≤ (P.constInputs.gates u).card * W := Circuit.width_le _ _ W hW
      _ ≤ (P.gates u).card * W := Nat.mul_le_mul_right _ (constInputs_bounds d P u).1
      _ ≤ Γ * W := Nat.mul_le_mul_right _ (hP u).1
  · obtain ⟨-, hunreg⟩ := (Verity.SecurityProofs.ExpandSize d.constInputs n P.constInputs).2 u
    have hx := hunreg.trans (constInputs_bounds d P u).2
    have hx' : ((d.constInputs.expand.unregistered (P.constInputs.expand.gates u)).card : ℝ) ≤
        novelReads d (P.gates u) := by
      exact_mod_cast hx
    refine floor_bound (hx'.trans (hP u).2) fun hpos => ?_
    have hW1 : 1 ≤ W := by
      by_contra h0
      have h00 : novelReads d (P.gates u) = 0 := by
        show ∑ g ∈ d.D.into (P.gates u) \ d.θ, d.D.w g = 0
        exact sum_eq_zero fun g _ => by have := hW g; omega
      exact absurd (lt_of_lt_of_le hpos (hx.trans_eq h00)) (lt_irrefl 0)
    have hΓ : Γ ≤ Γ * W :=
      calc Γ = Γ * 1 := (Nat.mul_one Γ).symm
        _ ≤ Γ * W := Nat.mul_le_mul_left Γ hW1
    exact (hP u).1.trans hΓ

end Nci.SecurityProofs
