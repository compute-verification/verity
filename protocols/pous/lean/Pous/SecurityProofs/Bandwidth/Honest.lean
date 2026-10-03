import Pous.Guarantees.Bandwidth

/-!
# The honest responder: `TwoTierBandwidth` holds of it, and the threshold is sharp

The honest responder reads the `R` challenged blocks in one round and returns their digest; its time is the host
link's time for the challenged blocks in `host`, the device tier free. It satisfies `TwoTierBandwidth`
(`honest_twoTierBandwidth`), so the assumption holds of some responder with `host` non-empty, and it is accepted
exactly when `b · h ≤ BWpcie · T` (`honest_accepts_iff`), where `HbmPlacement` takes `b · h > BWpcie · T`.
-/

namespace PousBandwidth

open Pous Pous.Bandwidth Pous.Assumptions
open scoped NNReal

variable {N R W : ℕ} {F : Type} [Field F] [Fintype F]

/-- The honest responder for the host-resident blocks `host`, at `b` bytes per block over a link of `BWpcie`. -/
noncomputable def honest (host : Finset (Fin N)) (b BWpcie : ℝ≥0) : BulkResponder N R W F where
  prog k S := .round (List.ofFn S) fun ans => .ret (digest k.1 k.2 fun r : Fin R => ans.getD r 0)
  time _ _ S := b * ((Finset.univ.image S ∩ host).card : ℝ≥0) / BWpcie

theorem honest_reads (host : Finset (Fin N)) (b BWpcie : ℝ≥0) (C : Store N W F) (k : Key F) (S : Fin R → Fin N) :
    reads C ((honest host b BWpcie).prog k S) = Finset.univ.image S := by
  ext i
  simp [honest, reads, List.mem_ofFn, eq_comm]

theorem honest_run (host : Finset (Fin N)) (b BWpcie : ℝ≥0) (C : Store N W F) (k : Key F) (S : Fin R → Fin N) :
    ((honest host b BWpcie).prog k S).run ((memory N W F).answer C) = digest k.1 k.2 (fun r => C (S r)) := by
  simp [honest, Prog.run, List.getD_eq_getElem?_getD]

theorem honest_twoTierBandwidth (host : Finset (Fin N)) (b BWpcie : ℝ≥0) (hBW : BWpcie ≠ 0) :
    TwoTierBandwidth host b BWpcie (honest (R := R) (W := W) (F := F) host b BWpcie) := by
  intro C k S
  rw [honest_reads]
  exact (mul_div_cancel₀ _ hBW).ge

theorem honest_accepts_iff (host : Finset (Fin N)) (b BWpcie T : ℝ≥0) (hBW : BWpcie ≠ 0) (S : Fin R → Fin N)
    (C : Store N W F) (k : Key F) :
    Accepts (honest host b BWpcie) T S C k ↔ b * ((Finset.univ.image S ∩ host).card : ℝ≥0) ≤ BWpcie * T := by
  unfold Accepts
  rw [honest_run, eq_self_iff_true, true_and]
  show b * _ / BWpcie ≤ T ↔ _
  rw [div_le_iff₀ (pos_iff_ne_zero.2 hBW), mul_comm T]

end PousBandwidth
