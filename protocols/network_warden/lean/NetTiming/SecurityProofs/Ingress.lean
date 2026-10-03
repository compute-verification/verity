import NetTiming.SecurityProofs.Advice

/-!
# The ingress timing bound

`docs/network-transparency/timing-channel.md` §2.5 ("Ingress content") and §3.2. The interaction is read inward
(`NetTiming.Protocol.Ingress`): the outside holds the message and reacts to every record, and a decoder inside X reads
the wire. On runs whose statuses are complete and whose frames follow the prescribed content, a record sequence is
determined by its occupancy counts and its jitter classes.
-/

namespace NetTiming

open Finset

variable {L K S : ℕ} {Msg In Act Wire : Type}

/-- **The ingress timing bound.** On runs whose statuses are complete up to `n` and whose frames on each link are a
prefix of the prescribed content, a decoder inside X that reads the wire tells apart at most
`(∏_ℓ (r ℓ + 1))^n · D^n` messages. -/
theorem encard_ingress_decodable_le (I : Interaction L K S Msg In Act Wire) (n D : ℕ) (r : Fin L → ℕ)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.warden view) (hW4 : Packed I.warden)
    (hr : Allocated I.warden r) (hJ : JitterBounded I.warden D)
    (c : Fin L → List (List Bool)) (Ms : Set Msg)
    (hacc : ∀ m ∈ Ms, StatusComplete n (I.obsRun m n))
    (hc : ∀ m ∈ Ms, ∀ ℓ, linkFrames (I.obsRun m n) ℓ <+: c ℓ)
    (dec : List Wire → Msg) (hdec : ∀ m ∈ Ms, dec (I.wireRun m n) = m) :
    Ms.encard ≤ (((∏ ℓ, (r ℓ + 1)) ^ n * D ^ n : ℕ) : ℕ∞) := by
  sorry

end NetTiming
