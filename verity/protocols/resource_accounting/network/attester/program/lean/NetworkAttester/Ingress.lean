module

public import NetworkAttester.Run

/-!
# The certifier program: acceptance as it bears on ingress

The executable form of `StatusComplete` from `security/Definitions/Certifier/Ingress.lean`, in `NetworkAttester`.
-/

@[expose] public section

namespace NetworkAttester

/-- **Acceptance, as it bears on ingress:** at every step before `k`, every boundary link's status is complete. -/
def StatusComplete {L K S : Nat} (k : Nat) (o : List (Obs L K S)) : Prop :=
  ∀ t (ht : t < k), ∃ h : t < o.length, ∀ κ, o[t].status κ = Status.complete

end NetworkAttester
