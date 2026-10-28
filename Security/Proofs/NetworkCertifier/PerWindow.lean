import Proofs.NetworkCertifier.Schedule
import Proofs.NetworkCertifier.Ingress

/-!
# Per window: the windows the accounting charges

`docs/network-transparency/timing-channel.md` states its bounds per sampling block. The guarantees count steps; these
read the horizon of the run's first `m` whole windows of `T` buckets, `n = m·T`, and factor each bound per window: the
shipped schedule's `#Σ_sync · D^T` (`encard_accSeqs_le_constantRate_windows` at `n = m·T`) and the ingress bound's
`(r ℓ + 1)^T` per sampling block (`encard_ingress_obsDecodable_le` and its content version at `n = m·T`). Each takes its
parent's hypotheses at that horizon and no other. The shipped schedule's factor is cumulative from the run's first
window: a later window may carry more than `#Σ_sync · D^T`.
-/

namespace NetTiming

open Finset

variable {L K S : ℕ} {Z Msg In Act Wire : Type}

/-- `Σ_sync` is never empty: its δs and its ρs both are not. -/
theorem SyncSet.size_pos (ss : SyncSet) : 0 < ss.size :=
  Nat.mul_pos (Finset.card_pos.mpr ss.deltas_nonempty) (Finset.card_pos.mpr ss.rhos_nonempty)

/-- The windows bound at `n = m·T`, read per window. -/
theorem windows_bound_le_perWindow (a s D T W m : ℕ) (hs : 0 < s) :
    a * s ^ min W m * D ^ (m * T) ≤ a * (s * D ^ T) ^ m := by
  rw [mul_pow, ← pow_mul, Nat.mul_comm T m, ← mul_assoc]
  exact Nat.mul_le_mul_right _ (Nat.mul_le_mul_left _ (Nat.pow_le_pow_right hs (min_le_right W m)))

/-- The ingress bound at `n = m·T`, read per sampling block. -/
theorem ingress_bound_eq_perWindow (r : Fin L → ℕ) (T m D : ℕ) :
    (∏ ℓ, (r ℓ + 1)) ^ (m * T) * D ^ (m * T) = ((∏ ℓ, (r ℓ + 1) ^ T) * D ^ T) ^ m := by
  rw [pow_mul', pow_mul', ← mul_pow, Finset.prod_pow]

theorem encard_accSeqs_le_constantRate_perWindow {Ac : Type} [Fintype Ac] (T W : ℕ) (ss : SyncSet)
    (frames : List In → Ac → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In) (m D : ℕ) :
    (AccSeqs (constantRate (S := S) T W ss frames) op (m * T) D).encard ≤
      ((Fintype.card Ac * (ss.size * D ^ T) ^ m : ℕ) : ℕ∞) :=
  (encard_accSeqs_le_constantRate_windows T W ss frames op (m * T) D m le_rfl).trans
    (Nat.cast_le.mpr (windows_bound_le_perWindow _ _ _ _ _ _ ss.size_pos))

theorem encard_decodable_le_constantRate_perWindow {Ac : Type} [Fintype Ac]
    (I : Interaction L K S Z In Act Wire) (T W : ℕ) (ss : SyncSet)
    (frames : List In → Ac → Fin L → List LogicalFrame) (m D : ℕ)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (Zs : Set Z)
    (hacc : ∀ z ∈ Zs, AcceptedUpTo (constantRate (S := S) T W ss frames) I.op (m * T) (I.obsRun z (m * T)))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z (m * T)) = z) :
    Zs.encard ≤ ((Fintype.card Ac * (ss.size * D ^ T) ^ m : ℕ) : ℕ∞) :=
  (encard_decodable_le_constantRate I T W ss frames (m * T) D m le_rfl view hW2 hJ Zs hacc dec hdec).trans
    (Nat.cast_le.mpr (windows_bound_le_perWindow _ _ _ _ _ _ ss.size_pos))

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
