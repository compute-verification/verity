module

public import Definitions.InferenceOnly

/-! An exact tiling at `(c, k)` complies at its rate; a block of fewer than `c` gates may read more than `k / c`
fresh input bits per gate. -/

@[expose] public section

namespace InferenceOnly.SecurityProofs

open Verity Finset

theorem compliesRate_of_complies {c k B : ℕ} {d : RegisteredCircuit} {P : Partition d.D B} (hP : P.IsCK d c k)
    (htile : ∀ u, (P.gates u).card = c) : CompliesRate c ((k : ℝ) / c) d := by
  refine ⟨B, P, fun u => ⟨(htile u).le, ?_⟩⟩
  have hw : freshInput d (P.gates u) = (d.unregistered (P.gates u)).card := by
    simp [freshInput, RegisteredCircuit.unregistered, Circuit.width, hP.boolean.one_bit]
  rw [hw, show d.D.size (P.gates u) = c from htile u]
  rcases Nat.eq_zero_or_pos c with rfl | hc
  · have h0 : P.gates u = ∅ := Finset.card_eq_zero.1 (htile u)
    simp [RegisteredCircuit.unregistered, Circuit.into, h0]
  · rw [div_mul_cancel₀ _ (Nat.cast_ne_zero.2 hc.ne')]
    exact_mod_cast hP.reads u

end InferenceOnly.SecurityProofs
