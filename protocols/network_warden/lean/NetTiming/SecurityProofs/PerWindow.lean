import NetTiming.SecurityProofs.Schedule
import NetTiming.SecurityProofs.Ingress

/-!
# Per window: the temporal unit the accounting charges

`docs/network-transparency/timing-channel.md` states its bounds per link-window. The guarantees count steps; these read
the horizon of `m` whole windows of `T` buckets, `n = m·T`, and factor each bound per window: the shipped schedule's
`#Σ_sync · D^T` (`encard_accSeqs_le_constantRate_windows` at `n = m·T`) and the ingress bound's `(r ℓ + 1)^T` per
link-window (`encard_ingress_obsDecodable_le` and its content version at `n = m·T`). Each takes its parent's hypotheses at
that horizon and no other.
-/

namespace NetTiming

open Finset

variable {L K S : ℕ} {Z Msg In Act Wire : Type}

theorem encard_accSeqs_le_constantRate_perWindow {Ac : Type} [Fintype Ac] (T W : ℕ) (ss : SyncSet)
    (frames : List In → Ac → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In) (m D : ℕ) :
    (AccSeqs (constantRate (S := S) T W ss frames) op (m * T) D).encard ≤
      ((Fintype.card Ac * (ss.size * D ^ T) ^ m : ℕ) : ℕ∞) := by
  sorry

theorem encard_decodable_le_constantRate_perWindow {Ac : Type} [Fintype Ac]
    (I : Interaction L K S Z In Act Wire) (T W : ℕ) (ss : SyncSet)
    (frames : List In → Ac → Fin L → List LogicalFrame) (m D : ℕ)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.warden view) (hJ : JitterBounded I.warden D)
    (Zs : Set Z)
    (hacc : ∀ z ∈ Zs, AcceptedUpTo (constantRate (S := S) T W ss frames) I.op (m * T) (I.obsRun z (m * T)))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z (m * T)) = z) :
    Zs.encard ≤ ((Fintype.card Ac * (ss.size * D ^ T) ^ m : ℕ) : ℕ∞) := by
  sorry

theorem encard_ingress_obsDecodable_le_perWindow (I : Interaction L K S Msg In Act Wire) (T m D : ℕ)
    (r : Fin L → ℕ) (hW4 : Packed I.warden) (hr : Allocated I.warden r) (hJ : JitterBounded I.warden D)
    (c : Fin L → List (List Bool)) (Ms : Set Msg)
    (hacc : ∀ msg ∈ Ms, StatusComplete (m * T) (I.obsRun msg (m * T)))
    (hc : ∀ msg ∈ Ms, ∀ ℓ, linkFrames (I.obsRun msg (m * T)) ℓ <+: c ℓ)
    (dec : List (Obs L K S) → Msg) (hdec : ∀ msg ∈ Ms, dec (I.obsRun msg (m * T)) = msg) :
    Ms.encard ≤ ((((∏ ℓ, (r ℓ + 1) ^ T) * D ^ T) ^ m : ℕ) : ℕ∞) := by
  sorry

theorem encard_ingress_obsDecodable_le_contents_perWindow (I : Interaction L K S Msg In Act Wire) (T m D : ℕ)
    (r : Fin L → ℕ) (hW4 : Packed I.warden) (hr : Allocated I.warden r) (hJ : JitterBounded I.warden D)
    (Cs : Set (Fin L → List (List Bool))) (Ms : Set Msg)
    (hacc : ∀ msg ∈ Ms, StatusComplete (m * T) (I.obsRun msg (m * T)))
    (hc : ∀ msg ∈ Ms, ∃ c ∈ Cs, ∀ ℓ, linkFrames (I.obsRun msg (m * T)) ℓ <+: c ℓ)
    (dec : List (Obs L K S) → Msg) (hdec : ∀ msg ∈ Ms, dec (I.obsRun msg (m * T)) = msg) :
    Ms.encard ≤ Cs.encard * ((((∏ ℓ, (r ℓ + 1) ^ T) * D ^ T) ^ m : ℕ) : ℕ∞) := by
  sorry

end NetTiming
