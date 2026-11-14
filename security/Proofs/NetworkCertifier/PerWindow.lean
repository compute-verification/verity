module

public import Definitions.NetworkCertifier.Assumptions
public import Definitions.NetworkCertifier.Ingress
public import Mathlib.Data.Set.Card
import Mathlib.Data.Sym.Sym2.Init
import Proofs.NetworkCertifier.Ingress

/-!
# Per window: the blocks the accounting charges

The ingress guarantees count steps; these count the monitoring period's whole periods of `T` buckets and factor the
ingress bound per period: `(r ℓ + 1)^T` per `T` buckets (`encard_ingress_obsDecodable_le` and its content version at
`n = m·T`). Each takes its parent's hypotheses at that length and no other.
-/

@[expose] public section

namespace NetTiming

open Finset

variable {L K S : ℕ} {Msg In Act Wire : Type}

/-- The ingress bound at `n = m·T`, read per link-window. -/
theorem ingress_bound_eq_perExecutionPhase (r : Fin L → ℕ) (T m D : ℕ) :
    (∏ ℓ, (r ℓ + 1)) ^ (m * T) * D ^ (m * T) = ((∏ ℓ, (r ℓ + 1) ^ T) * D ^ T) ^ m := by
  rw [pow_mul', pow_mul', ← mul_pow, Finset.prod_pow]

theorem encard_ingress_obsDecodable_le_perExecutionPhase (I : Interaction L K S Msg In Act Wire) (T m D : ℕ)
    (r : Fin L → ℕ) (hW4 : Packed I.certifier) (hr : Allocated I.certifier r) (hJ : JitterBounded I.certifier D)
    (c : Fin L → List (List Bool)) (Ms : Set Msg)
    (hacc : ∀ msg ∈ Ms, StatusComplete (m * T) (I.obsRun msg (m * T)))
    (hc : ∀ msg ∈ Ms, ∀ ℓ, linkFrames (I.obsRun msg (m * T)) ℓ <+: c ℓ)
    (dec : List (Obs L K S) → Msg) (hdec : ∀ msg ∈ Ms, dec (I.obsRun msg (m * T)) = msg) :
    Ms.encard ≤ ((((∏ ℓ, (r ℓ + 1) ^ T) * D ^ T) ^ m : ℕ) : ℕ∞) := by
  rw [← ingress_bound_eq_perExecutionPhase]
  exact encard_ingress_obsDecodable_le I (m * T) D r hW4 hr hJ c Ms hacc hc dec hdec

theorem encard_ingress_obsDecodable_le_contents_perExecutionPhase (I : Interaction L K S Msg In Act Wire) (T m D : ℕ)
    (r : Fin L → ℕ) (hW4 : Packed I.certifier) (hr : Allocated I.certifier r) (hJ : JitterBounded I.certifier D)
    (Cs : Set (Fin L → List (List Bool))) (Ms : Set Msg)
    (hacc : ∀ msg ∈ Ms, StatusComplete (m * T) (I.obsRun msg (m * T)))
    (hc : ∀ msg ∈ Ms, ∃ c ∈ Cs, ∀ ℓ, linkFrames (I.obsRun msg (m * T)) ℓ <+: c ℓ)
    (dec : List (Obs L K S) → Msg) (hdec : ∀ msg ∈ Ms, dec (I.obsRun msg (m * T)) = msg) :
    Ms.encard ≤ Cs.encard * ((((∏ ℓ, (r ℓ + 1) ^ T) * D ^ T) ^ m : ℕ) : ℕ∞) := by
  rw [← ingress_bound_eq_perExecutionPhase]
  exact encard_ingress_obsDecodable_le_contents I (m * T) D r hW4 hr hJ Cs Ms hacc hc dec hdec

end NetTiming
