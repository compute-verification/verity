module

public import Proofs.Properties.Nci.Record

/-!
# The rate rule at its edges

A scratch file, never merged. What `Rates`, `Counts` and `Complies` say where a parameter is zero or a block is empty.

* `novelReads_empty`: a block of no gate reads nothing, so `Counts`' second case asks nothing of the summary's fresh
  bits.
* `rates_gamma_zero`, `rates_num_zero`: at `Γ = 0` the summary has no gate and no fresh bit over a positive `den`; at
  `num = 0` it has no fresh bit over a positive `den`.
* `alpha_den_zero`: at `den = 0` the rate `num / den` is zero, where `Rates`' second bound says nothing
  (`rates_den_zero`): only `Carries`' `0 < den` keeps `Complies` from asking more than `Rates` gives.
* `complies_gamma_zero`: where a check accepts at `Γ = 0`, every circuit the record lays out has no computed gate in
  any block.
-/

@[expose] public section

namespace NciCheck.Properties.Record.Attack

open Lean (Json)

/-- **A block of no gate reads nothing.** -/
theorem novelReads_empty (d : Verity.RegisteredCircuit) : Nci.novelReads d ∅ = 0 := by
  simp [Nci.novelReads, Verity.Circuit.into, Verity.Circuit.width]

/-- At `Γ = 0` a summary the rate rule passes has no gate, and no fresh bit over a positive `den`. -/
theorem rates_gamma_zero {F : Form} (hr : Rates F) {num den : ℕ} (hden : 0 < den) {s : Summary}
    (h : F.rate 0 num den s = none) : s.gates = 0 ∧ s.fresh = 0 := by
  obtain ⟨hg, hf⟩ := hr 0 num den s h
  have hg0 : s.gates = 0 := Nat.le_zero.mp hg
  rw [hg0, Nat.mul_zero] at hf
  have hf0 : s.fresh * den = 0 := Nat.le_zero.mp hf
  rcases Nat.mul_eq_zero.mp hf0 with h0 | h0
  · exact ⟨hg0, h0⟩
  · exact absurd h0 (Nat.ne_of_gt hden)

/-- At `num = 0` a summary the rate rule passes has no fresh bit over a positive `den`. -/
theorem rates_num_zero {F : Form} (hr : Rates F) {Γ den : ℕ} (hden : 0 < den) {s : Summary}
    (h : F.rate Γ 0 den s = none) : s.fresh = 0 := by
  obtain ⟨-, hf⟩ := hr Γ 0 den s h
  rw [Nat.zero_mul] at hf
  have hf0 : s.fresh * den = 0 := Nat.le_zero.mp hf
  rcases Nat.mul_eq_zero.mp hf0 with h0 | h0
  · exact h0
  · exact absurd h0 (Nat.ne_of_gt hden)

/-- At `den = 0` the second bound of `Rates` holds of every summary. -/
theorem rates_den_zero (num : ℕ) (s : Summary) : s.fresh * 0 ≤ num * s.gates := by
  rw [Nat.mul_zero]
  exact Nat.zero_le _

/-- At `den = 0` the rate `Complies` states is zero. -/
theorem alpha_den_zero (num : ℕ) : ((num : ℝ) / ((0 : ℕ) : ℝ)) = 0 := by simp

/-- Where a check accepts at `Γ = 0`, no block of a circuit the record lays out holds a gate. -/
theorem complies_gamma_zero {F : Form} (hc : Complies F) {j tr : Json} {c : F.Case} (hacc : Accepts F j tr c)
    (hΓ : (F.inst c).gamma = 0) {d : Verity.RegisteredCircuit} (hd : Served F c d) (u : Fin d.n) :
    d.𝓘.gates u = ∅ := by
  obtain ⟨-, -, hall⟩ := hc j tr c hacc
  obtain ⟨hmeets, -⟩ := hall d hd
  obtain ⟨hsize, -⟩ := hmeets u
  rw [hΓ] at hsize
  have h0 : (d.𝓘.gates u).card = 0 := Nat.le_zero.mp hsize
  exact Finset.card_eq_zero.mp h0

end NciCheck.Properties.Record.Attack
