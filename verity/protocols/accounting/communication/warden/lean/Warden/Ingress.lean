import Warden.Run

/-!
# The warden program: acceptance as it bears on ingress

The executable form of `StatusComplete` from `verity/Security/Definitions/Warden/Ingress.lean`, in `NetTiming.Code`.
-/

namespace NetTiming.Code

/-- **Acceptance, as it bears on ingress:** at every step before `k`, every boundary link's status is complete. -/
def StatusComplete {L K S : Nat} (k : Nat) (o : List (Obs L K S)) : Prop :=
  ∀ t (ht : t < k), ∃ h : t < o.length, ∀ κ, o[t].status κ = Status.complete

end NetTiming.Code
