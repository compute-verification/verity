module

public import Definitions.Nci

/-!
# An exact tiling at `(c, k)` complies at its rate

An inference block of a `(c, k)`-partition has at most `c` gates and at most `k` fresh input bits, each gate one bit
wide. A block of fewer than `c` gates may read more than `k / c` fresh input bits per gate, so only a partition whose
blocks have exactly `c` gates complies at `c` gates per block and the rate `k / c` (`compliesRate_of_complies`, with the
tiling as its hypothesis). The guarantees are stated over `CompliesRate`, the form the inference-only check's verdicts
read.
-/

@[expose] public section

namespace Nci.SecurityProofs

open Verity Finset

theorem compliesRate_of_complies {c k B : ℕ} {d : RegisteredCircuit} {P : Partition d.D B} (hP : P.IsCK d c k)
    (htile : ∀ u, (P.gates u).card = c) : CompliesRate c ((k : ℝ) / c) d := by
  refine ⟨B, P, fun u => ⟨(htile u).le, ?_⟩⟩
  have hw : novelReads d (P.gates u) = (d.unregistered (P.gates u)).card := by
    simp [novelReads, RegisteredCircuit.unregistered, Circuit.width, hP.boolean.one_bit]
  rw [hw, show d.D.size (P.gates u) = c from htile u]
  rcases Nat.eq_zero_or_pos c with rfl | hc
  · have h0 : P.gates u = ∅ := Finset.card_eq_zero.1 (htile u)
    simp [RegisteredCircuit.unregistered, Circuit.into, h0]
  · rw [div_mul_cancel₀ _ (Nat.cast_ne_zero.2 hc.ne')]
    exact_mod_cast hP.reads u

end Nci.SecurityProofs
