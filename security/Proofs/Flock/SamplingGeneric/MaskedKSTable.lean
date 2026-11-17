module

public import Proofs.Flock.SamplingGeneric.Masked
public import Proofs.Flock.SamplingGeneric.KSTable
public import Proofs.Flock.Claims.KS.ZkKSTable

/-! The sampled compiler over Flock's deployed masked record at one table of a session, at SHA-512. -/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims.Sampling (Opening pathCommit hm96Leaf regPath RowPlaces pathCommit_binds hm96Leaf_binds regPath_binds
  listWithin_deployZk)

variable (names : Msg → ℕ × Msg)
  (L : Compiled FlockDeploy.Coin
    (Arg.ofFlockTable (FlockDeploy.deployZk _root_.Flock.Readable.relation) (FlockDeploy.KS.wordLenJ true)
      (FlockDeploy.KS.tableErrJ true) names))
  {own : Array _root_.Flock.Registered.Port} {at_ : List Msg → ℕ → Option (Msg × ℕ)}
  {row salt π : Opening Bool → Msg} {layer : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg}

theorem ksTable_masked (hcommit : L.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer)
    (hrow : RowPlaces row) (hS : L.Sessions FlockDeploy.sha512) : L.KS FlockDeploy.sha512 := by
  refine flockKSTable FlockDeploy.sha512 _ _ _ names L FlockDeploy.KS.zk_KSTable_untrim (listWithin_deployZk _) ?_ hS
  rw [hcommit]
  exact pathCommit_binds (hm96Leaf_binds _ hrow salt) (regPath_binds _ own)

theorem ksRegTable_masked (hcommit : L.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer)
    (hrow : RowPlaces row) (hS : L.Sessions FlockDeploy.sha512) : L.KSReg FlockDeploy.sha512 := by
  refine flockKSRegTable FlockDeploy.sha512 _ _ _ names L FlockDeploy.KS.zk_KSTable_untrim (listWithin_deployZk _) ?_ hS
  rw [hcommit]
  exact pathCommit_binds (hm96Leaf_binds _ hrow salt) (regPath_binds _ own)

end Claims.SamplingGeneric
