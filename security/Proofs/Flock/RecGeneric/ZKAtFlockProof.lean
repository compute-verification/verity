module

public import Proofs.Flock.RecGeneric.ZKAtFlock
public import Proofs.Flock.RecGeneric.ZKSessionsProof
public import Proofs.Flock.RecGeneric.ZKHides

/-!
# Z1-F's proof

Z1 at sessions (`Flock.SecurityProofs.GenericRecursiveSessionsZK`) at hm96 leaves, whose hiding is Z2
(`Flock.SecurityProofs.Hm96Hides`), and at Flock's ZK server's session system, whose zero knowledge is Z1-L
(`Flock.SecurityProofs.GenericSessionsOfZKOut`) at `FlockZk.server_zkOut`.
-/

@[expose] public section

open Claims Claims.RecGeneric Claims.SamplingGeneric in
set_option synthInstance.maxSize 100000 in
set_option synthInstance.maxHeartbeats 1000000 in
/-- **Z1-F** (`Flock.Guarantees.GenericRecursiveZKAtFlockServer`). -/
theorem Flock.SecurityProofs.GenericRecursiveZKAtFlockServer : Flock.Guarantees.GenericRecursiveZKAtFlockServer := by
  intro rel T vs t δ₁ ht hT1 hS hV hD hc hsc hT L₀ L hL wk pc fc Xs enc W F Ls hSh
  exact Flock.SecurityProofs.GenericRecursiveSessionsZK FlockZk.sha (Leaves.hm96 enc) _ Ls F _ (δ₁ : ℝ)
    (Flock.SecurityProofs.Hm96Hides enc δ₁ hT1)
    (Flock.SecurityProofs.GenericSessionsOfZKOut FlockZk.sha _ (FlockZk.rounds rel) (FlockZk.Tape T) L.toSrv.Chk _ _ _
      Xs _ (FlockZk.server_zkOut rel T vs t ht hT1 ENNReal.coe_ne_top hS hV hD hc hsc hT L₀ L hL wk pc fc))
    hSh
