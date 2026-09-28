import NetworkCertifier.Run

/-!
# The certifier program: acceptance as it bears on ingress

The executable form of `StatusComplete` from `Security/Definitions/NetworkCertifier/Ingress.lean`, in `NetworkCertifier`.
-/

namespace NetworkCertifier

/-- **Acceptance, as it bears on ingress:** at every step before `k`, every boundary link's status is complete. -/
def StatusComplete {L K S : Nat} (k : Nat) (o : List (Obs L K S)) : Prop :=
  ∀ t (ht : t < k), ∃ h : t < o.length, ∀ κ, o[t].status κ = Status.complete

end NetworkCertifier
