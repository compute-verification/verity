module

public import Proofs.Flock.SamplingGeneric.Masked
public import Proofs.Flock.SamplingGeneric.TableAt
public import Proofs.Flock.Claims.KS.ZkTableAt

/-!
# The sampled compiler over Flock's masked record at one table of a session, at SHA-512

`flockKSAt` and `flockKSRegAt` hold at every Flock record, hash, tables' lengths and errors, taking `Flock.KSAt` there,
the decoder's list bound and the commitment's binding as hypotheses. At Flock's deployed masked record
(`FlockDeploy.deployZk` at the deployed relation), its tables' words' lengths, the tables' errors summed and SHA-512,
the three are proved: `Flock.KSAt` is `FlockDeploy.KS.zk_KSAt_untrim`, the list bound `listWithin_deployZk`, and hm96
leaves of rows that place the value, under the registered path check, bind at every hash (`pathCommit_binds`). So a
compiled verifier over Flock at one table of a session there, with such a commitment, is knowledge sound, in both
forms, wherever it meets the contract (`ksAt_masked`, `ksRegAt_masked`), at every way its statements name a table and
a Flock statement. Its sessions' expected reruns are one table's (`flockRerunsAt`).
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims.Sampling (Opening pathCommit hm96Leaf regPath RowPlaces pathCommit_binds hm96Leaf_binds regPath_binds
  listWithin_deployZk)

variable (names : Msg → ℕ × Msg)
  (L : Compiled FlockDeploy.Coin
    (Arg.ofFlockAt (FlockDeploy.deployZk _root_.Flock.Readable.relation) (FlockDeploy.KS.wordLenJ true)
      (fun x _ => (FlockDeploy.deployZk _root_.Flock.Readable.relation).εtable x) names))
  {own : Array _root_.Flock.Registered.Port} {at_ : List Msg → ℕ → Option (Msg × ℕ)}
  {row salt π : Opening Bool → Msg} {layer : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg}

/-- **The compiler over Flock's masked record at one table of a session is knowledge sound at SHA-512**: a compiled
verifier over it at its deployed decoder, committing by hm96 leaves of rows that place the value under the registered
path check, that meets the contract (`flockKSAt`: `Flock.KSAt` by `zk_KSAt_untrim`, the list bound by
`listWithin_deployZk`, binding by `pathCommit_binds`). -/
theorem ksAt_masked (hcommit : L.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer)
    (hrow : RowPlaces row) (hS : L.Sessions FlockDeploy.sha512) : L.KS FlockDeploy.sha512 := by
  refine flockKSAt FlockDeploy.sha512 _ _ _ names L FlockDeploy.KS.zk_KSAt_untrim (listWithin_deployZk _) ?_ hS
  rw [hcommit]
  exact pathCommit_binds (hm96Leaf_binds _ hrow salt) (regPath_binds _ own)

/-- **Its committed form**: as `ksAt_masked`, `KSReg` (`flockKSRegAt`). -/
theorem ksRegAt_masked (hcommit : L.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer)
    (hrow : RowPlaces row) (hS : L.Sessions FlockDeploy.sha512) : L.KSReg FlockDeploy.sha512 := by
  refine flockKSRegAt FlockDeploy.sha512 _ _ _ names L FlockDeploy.KS.zk_KSAt_untrim (listWithin_deployZk _) ?_ hS
  rw [hcommit]
  exact pathCommit_binds (hm96Leaf_binds _ hrow salt) (regPath_binds _ own)

end Claims.SamplingGeneric
