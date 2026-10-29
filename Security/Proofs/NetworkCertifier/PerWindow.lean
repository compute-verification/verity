import Proofs.NetworkCertifier.Schedule
import Proofs.NetworkCertifier.Ingress

/-!
# Per window: the blocks the accounting charges

`docs/network-transparency/timing-channel.md` states its bounds per link-window. The guarantees count steps; these read
a horizon of whole blocks and factor each bound per block: the shipped schedule's `2^k · D^(g·T)` per execution block
(`encard_accSeqs_le_constantRate_windows_bits` at `n = m·(g·T)`), and the ingress bound's `(r ℓ + 1)^T` per `T` buckets
(`encard_ingress_obsDecodable_le` and its content version at `n = m·T`). Each takes its parent's hypotheses at that
horizon and no other.
-/

namespace NetTiming

open Finset

variable {L K S : ℕ} {Z Msg In Act Wire : Type}

/-- The blocks bound at `n = m·N`, read per execution block of `N` buckets. -/
theorem blocks_bound_le_perWindow (k D N E m : ℕ) :
    2 ^ (min E m * k) * D ^ (m * N) ≤ (2 ^ k * D ^ N) ^ m := by
  rw [mul_pow, ← pow_mul', ← pow_mul']
  exact Nat.mul_le_mul_right _ (Nat.pow_le_pow_right Nat.two_pos (Nat.mul_le_mul_right _ (min_le_right E m)))

/-- The ingress bound at `n = m·T`, read per link-window. -/
theorem ingress_bound_eq_perWindow (r : Fin L → ℕ) (T m D : ℕ) :
    (∏ ℓ, (r ℓ + 1)) ^ (m * T) * D ^ (m * T) = ((∏ ℓ, (r ℓ + 1) ^ T) * D ^ T) ^ m := by
  rw [pow_mul', pow_mul', ← mul_pow, Finset.prod_pow]

theorem encard_accSeqs_le_constantRate_perWindow (T g E k : ℕ)
    (frames : List In → Fin E → (Fin k → Bool) → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In)
    (m D : ℕ) :
    (AccSeqs (constantRate (S := S) T g E frames) op (m * (g * T)) D).encard ≤
      (((2 ^ k * D ^ (g * T)) ^ m : ℕ) : ℕ∞) :=
  (encard_accSeqs_le_constantRate_windows_bits T g E k frames op (m * (g * T)) D m le_rfl).trans
    (Nat.cast_le.mpr (blocks_bound_le_perWindow _ _ _ _ _))

theorem encard_decodable_le_constantRate_perWindow
    (I : Interaction L K S Z In Act Wire) (T g E k : ℕ)
    (frames : List In → Fin E → (Fin k → Bool) → Fin L → List LogicalFrame) (m D : ℕ)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (Zs : Set Z)
    (hacc : ∀ z ∈ Zs, AcceptedUpTo (constantRate (S := S) T g E frames) I.op (m * (g * T))
      (I.obsRun z (m * (g * T))))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z (m * (g * T))) = z) :
    Zs.encard ≤ (((2 ^ k * D ^ (g * T)) ^ m : ℕ) : ℕ∞) :=
  (encard_decodable_le_constantRate_bits I T g E k frames (m * (g * T)) D m le_rfl view hW2 hJ Zs hacc dec
    hdec).trans (Nat.cast_le.mpr (blocks_bound_le_perWindow _ _ _ _ _))

theorem encard_ingress_obsDecodable_le_perWindow (I : Interaction L K S Msg In Act Wire) (T m D : ℕ)
    (r : Fin L → ℕ) (hW4 : Packed I.certifier) (hr : Allocated I.certifier r) (hJ : JitterBounded I.certifier D)
    (c : Fin L → List (List Bool)) (Ms : Set Msg)
    (hacc : ∀ msg ∈ Ms, StatusComplete (m * T) (I.obsRun msg (m * T)))
    (hc : ∀ msg ∈ Ms, ∀ ℓ, linkFrames (I.obsRun msg (m * T)) ℓ <+: c ℓ)
    (dec : List (Obs L K S) → Msg) (hdec : ∀ msg ∈ Ms, dec (I.obsRun msg (m * T)) = msg) :
    Ms.encard ≤ ((((∏ ℓ, (r ℓ + 1) ^ T) * D ^ T) ^ m : ℕ) : ℕ∞) := by
  rw [← ingress_bound_eq_perWindow]
  exact encard_ingress_obsDecodable_le I (m * T) D r hW4 hr hJ c Ms hacc hc dec hdec

theorem encard_ingress_obsDecodable_le_contents_perWindow (I : Interaction L K S Msg In Act Wire) (T m D : ℕ)
    (r : Fin L → ℕ) (hW4 : Packed I.certifier) (hr : Allocated I.certifier r) (hJ : JitterBounded I.certifier D)
    (Cs : Set (Fin L → List (List Bool))) (Ms : Set Msg)
    (hacc : ∀ msg ∈ Ms, StatusComplete (m * T) (I.obsRun msg (m * T)))
    (hc : ∀ msg ∈ Ms, ∃ c ∈ Cs, ∀ ℓ, linkFrames (I.obsRun msg (m * T)) ℓ <+: c ℓ)
    (dec : List (Obs L K S) → Msg) (hdec : ∀ msg ∈ Ms, dec (I.obsRun msg (m * T)) = msg) :
    Ms.encard ≤ Cs.encard * ((((∏ ℓ, (r ℓ + 1) ^ T) * D ^ T) ^ m : ℕ) : ℕ∞) := by
  rw [← ingress_bound_eq_perWindow]
  exact encard_ingress_obsDecodable_le_contents I (m * T) D r hW4 hr hJ Cs Ms hacc hc dec hdec

end NetTiming
