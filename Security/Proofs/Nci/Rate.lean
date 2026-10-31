import Definitions.Nci

/-!
# The policy at `(c, k)` complies at its rate

An inference block of a `(c, k)`-partition has exactly `c` gates and at most `k` fresh input bits, each gate one bit
wide, so the partition complies at `c` gates per block and the rate `k / c` (`compliesRate_of_complies`), and at
`c > 0` the rate times `c` is `k` (`rate_mul`). Each guarantee over `Complies c k` is proved from its rate form over
`CompliesRate`, the form the inference-only check's verdicts read.
-/

namespace Nci.SecurityProofs

open Verity Finset

theorem compliesRate_of_complies {c k : ℕ} {d : RegisteredCircuit} (h : Complies c k d) :
    CompliesRate c ((k : ℝ) / c) d := by
  obtain ⟨B, P, hP⟩ := h
  refine ⟨B, P, fun u => ⟨(hP.card u).le, ?_⟩⟩
  have hw : novelReads d (P.gates u) = (d.unregistered (P.gates u)).card := by
    simp [novelReads, RegisteredCircuit.unregistered, Circuit.width, hP.boolean.one_bit]
  rw [hw, show d.D.size (P.gates u) = c from hP.card u]
  rcases Nat.eq_zero_or_pos c with rfl | hc
  · have h0 : P.gates u = ∅ := Finset.card_eq_zero.1 (hP.card u)
    simp [RegisteredCircuit.unregistered, Circuit.into, h0]
  · rw [div_mul_cancel₀ _ (Nat.cast_ne_zero.2 hc.ne')]
    exact_mod_cast hP.reads u

theorem rate_mul {c : ℕ} (k : ℝ) (hc : 0 < c) : k / c * c = k :=
  div_mul_cancel₀ _ (Nat.cast_ne_zero.2 hc.ne')

end Nci.SecurityProofs
