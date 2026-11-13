module

public import Proofs.Flock.SamplingGeneric.Masked
public import Proofs.Flock.SamplingGeneric.KSTable
public import Proofs.Flock.Claims.KS.ZkKSTable

/-!
# The sampled compiler over Flock's masked record at one table of a session, at SHA-512

`flockKSTable` and `flockKSRegTable` hold at every Flock record, hash, tables' lengths and errors, taking
`Flock.KSTable` there, the decoder's list bound and the commitment's binding as hypotheses. At Flock's deployed masked
record (`FlockDeploy.deployZk` at the deployed relation), its tables' words' lengths, each table's own error
(`FlockDeploy.KS.tableErrJ`) and SHA-512, the three are proved: `Flock.KSTable` is `FlockDeploy.KS.zk_KSTable_untrim`,
the list bound `listWithin_deployZk`, and hm96 leaves of rows that place the value, under the registered path check,
bind at every hash (`pathCommit_binds`). So a compiled verifier over Flock at one table of a session there, with such a
commitment, is knowledge sound, in both forms, wherever it meets the contract (`ksTable_masked`, `ksRegTable_masked`),
at every way its statements name a table and a Flock statement. Each wait of its extractor stops at one table's `k_j`
(`flockRerunsTable`), and its knowledge error reads one table's error.
-/

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

/-- **The compiler over Flock's masked record at one table of a session is knowledge sound at SHA-512**: a compiled
verifier over it at its deployed decoder, committing by hm96 leaves of rows that place the value under the registered
path check, that meets the contract (`flockKSTable`: `Flock.KSTable` by `zk_KSTable_untrim`, the list bound by
`listWithin_deployZk`, binding by `pathCommit_binds`). -/
theorem ksTable_masked (hcommit : L.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer)
    (hrow : RowPlaces row) (hS : L.Sessions FlockDeploy.sha512) : L.KS FlockDeploy.sha512 := by
  refine flockKSTable FlockDeploy.sha512 _ _ _ names L FlockDeploy.KS.zk_KSTable_untrim (listWithin_deployZk _) ?_ hS
  rw [hcommit]
  exact pathCommit_binds (hm96Leaf_binds _ hrow salt) (regPath_binds _ own)

/-- **Its committed form**: as `ksTable_masked`, `KSReg` (`flockKSRegTable`). -/
theorem ksRegTable_masked (hcommit : L.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer)
    (hrow : RowPlaces row) (hS : L.Sessions FlockDeploy.sha512) : L.KSReg FlockDeploy.sha512 := by
  refine flockKSRegTable FlockDeploy.sha512 _ _ _ names L FlockDeploy.KS.zk_KSTable_untrim (listWithin_deployZk _) ?_ hS
  rw [hcommit]
  exact pathCommit_binds (hm96Leaf_binds _ hrow salt) (regPath_binds _ own)

end Claims.SamplingGeneric
