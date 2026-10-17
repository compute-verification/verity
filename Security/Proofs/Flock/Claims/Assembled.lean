import Proofs.Flock.Claims.Bits
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.DeployEfficient
import Proofs.Flock.Claims.Complete.WalkPacked
import Proofs.Flock.Claims.Complete.ZkTables.StmtOne
import Proofs.Flock.Soundness.CompiledSound

/-!
# Flock's claims at the deployed records, with what remains as hypotheses

`flock_plain_assembled` and `flock_zk_assembled`: at `FlockDeploy.deployPlain rel` and `FlockDeploy.deployZk rel`, at
every relation's fields `rel` and every hash `H`, the conjunction of Flock's claims:

* `ScopeMatches`: the readable scope is the verifier's (`scopeMatches_plain`, `scopeMatches_zk`);
* `KS` at `E^rep` and every decoder meeting `ListWithin`: a hypothesis;
* `Bits` at every such decoder (`Flock.Bits_of_KS`), and at the radius decoder `min(a, 255.5 − b/2) − 1` bits at every
  bound `κ ≤ 2^{−a}`, `γ ≤ 2^b` (`Flock.Bits_ge`), from `KS` with `ε_table ≥ 0` and `γ > 0` (`εtable_nonneg`,
  `γ_pos`);
* `Efficient` at the record's own counted decoder (`GS.plainCountedRD`, `GS.zkCountedRD`): `E^rep`'s reruns at every
  deployment (`ERep_cost`), the decoder's `ListWithin`, and its count and lists at most `plainStepPoly` (`zkStepPoly`)
  at the opened word's encoded length, which bounds `assemble`'s checks (`efficient_plain`, `efficient_zk`);
* `Complete` at the record's reference prover (`honestP`, `ZkTables.honestP`) at the record's error (`δc`,
  `ZkTables.δc`): a hypothesis, which at the deployed relation follows from the record's JSON at the plain record
  (`flock_plain_assembled_relation`), and at the masked record from it and three premises on the honest masked run
  (`flock_zk_assembled_relation`).

## What remains: the hypotheses

* `hKS : Flock.KS H (deployPlain rel)`, and at `deployZk rel`: E4, `cursor/claims-ks-c7a9`. There KS at SHA-512 rests
  on `TableBoundθ` at the statements that set up (`plain_KSθ_of_setup`, `zk_KSθ_of_setup`; `ConflictTied` and
  `ColsOk` are proved there), but it concludes the branch's radius form `KSθ`, not `Flock.KS`. `KSθ`'s extractor
  decodes the word filled to table 0's level-0 length with a zero column of the lanes' width (`Eθ`, `fillWord`,
  `zeroCol`), at every decoder meeting `ListCompleteθ`; `Flock.KS`'s decodes the opened columns, at every decoder
  meeting `ListWithin`, whose word has `[]` where no column was opened. So the hypothesis is `Flock.KS` itself.
* `hComplete : Flock.Complete H (deployPlain rel)`, of the record's reference prover `honestP` at its error
  `FlockComplete.δc H`: at the deployed relation (`Readable.relation`) it follows from the record's JSON
  (`Claims.JsonSpec printJ canon`, `FlockComplete.Replay.complete_plain_json`), `Out64` being `Complete`'s own
  antecedent, so `flock_plain_assembled_relation` takes `JsonSpec` in its place. At any other `rel` it stays a
  hypothesis: `rel` gives the readable statement's circuit, which the relation's witnesses satisfy, and where it is not
  the verifier's tables' the honest prover's trace need not satisfy them.
* `hComplete : Flock.Complete H (deployZk rel)`, of the masked record's reference prover `FlockComplete.ZkTables.honestP`
  (each table run on its share of the witness) at its error `FlockComplete.ZkTables.δc H`, at any number of tables: at
  the deployed relation it follows from the record's JSON and three premises on the honest masked run, each at every
  good case or run and every table of it (`FlockComplete.ZkTables.complete_json`), so `flock_zk_assembled_relation`
  takes those in its place: the masked prover's pieces return (`ProverRunsAt`), its rep proofs fit their shapes
  (`RepShapedAt`) and, per rep, the masked Ligerito and the inner proof accept (`LgInnerAt`). Those three follow from
  four premises on one rep of one table (`FlockComplete.ZkTables.complete_jsonT`: `RepShapedT`, `ProverRunsT`,
  `LigeritoT`, `InnerT`). At any other `rel` it stays a hypothesis, as at `deployPlain rel`.
-/

namespace FlockDeploy.Assembled

open Flock

/-- A table's error is nonnegative. -/
theorem tableErr_nonneg (zk : Bool) (spec : SessionSpec) (st : Stmt) : 0 ≤ tableErr zk spec st := by
  unfold tableErr
  split
  · unfold FlockSoundness.Accounting.padLinkError
    positivity
  · exact FlockSoundness.tableError_nonneg _ _

/-- **`ε_table ≥ 0`** at both records: the tables' errors, summed. -/
theorem εtable_nonneg (zk : Bool) (x : Claims.Msg) : 0 ≤ εtable zk x := by
  unfold εtable
  refine List.sum_nonneg fun a ha => ?_
  obtain ⟨p, -, rfl⟩ := List.mem_map.1 ha
  exact tableErr_nonneg zk p.1 p.2

/-- The deployed verifier takes at least one round at every statement. -/
theorem V_k_pos (zk : Bool) (x : Claims.Msg) : 0 < (V zk).k x := by
  simp only [V]
  split <;> omega

/-- **`γ > 0`** wherever the verifier takes a round. -/
theorem γ_pos {Coin Col T : Type} (D : Claims.Flock.Deploy Coin Col T) (x : Claims.Msg) (hk : 0 < D.V.k x) :
    0 < Claims.Flock.γ D x := by
  unfold Claims.Flock.γ
  have hk' : (0 : ℝ) < D.V.k x := by exact_mod_cast hk
  have hs := Real.sqrt_nonneg (2 * D.V.k x * D.weight x)
  have h : 0 < Real.sqrt (2 * D.V.k x * D.weight x) + D.V.k x := by linarith
  exact mul_pos (by norm_num) (pow_pos h 2)

/-- **Flock's claims at the deployed plain verifier**, at every relation's fields and every hash: the readable scope
is the verifier's; KS at `E^rep`; bits at every decoder meeting `ListWithin`, and at the radius decoder at every bound
on `κ` and `γ`; efficiency at the radius decoder with its count; and completeness at the record's reference prover.
`hKS` (E4) and `hComplete` (C3) are what remains. -/
theorem flock_plain_assembled (rel : Readable.Parsed → Readable.Relation) (H : Claims.Msg → Claims.Msg)
    (hKS : Claims.Flock.KS H (deployPlain rel))
    (hComplete : Claims.Flock.Complete H (deployPlain rel)) :
    Claims.Flock.ScopeMatches (deployPlain rel) ∧
      Claims.Flock.KS H (deployPlain rel) ∧
      Claims.Flock.Bits H (deployPlain rel) ∧
      (∀ a b : Claims.Msg → ℝ, (∀ x, Claims.Flock.κ (deployPlain rel) x ≤ 2 ^ (-a x)) →
        (∀ x, Claims.Flock.γ (deployPlain rel) x ≤ 2 ^ b x) →
        Claims.Bits H (deployPlain rel).V (Claims.Flock.R (deployPlain rel).parse)
          (Claims.Flock.E (deployPlain rel) Claims.GS.plainDecodeRD) fun x => min (a x) (255.5 - b x / 2) - 1) ∧
      Claims.Flock.Efficient H (deployPlain rel) plainStepPoly ∧
      Claims.Flock.Complete H (deployPlain rel) :=
  ⟨scopeMatches_plain rel, hKS, Claims.Flock.Bits_of_KS hKS (εtable_nonneg false),
    fun _ _ hκ hγ => Claims.Flock.Bits_ge hKS (εtable_nonneg false) hκ
      (fun x => γ_pos (deployPlain rel) x (V_k_pos false x)) hγ _ (listWithin_plain rel),
    efficient_plain H rel, hComplete⟩

/-- **Flock's claims at the deployed plain verifier at the deployed relation** (`Readable.relation`), at every hash:
`flock_plain_assembled` with its completeness from the record's JSON (`complete_plain_json`). `hKS` (E4) is what
remains. -/
theorem flock_plain_assembled_relation (H : Claims.Msg → Claims.Msg)
    (hKS : Claims.Flock.KS H (deployPlain Readable.relation))
    (S : Claims.JsonSpec FlockComplete.printJ Flock.canon) :
    Claims.Flock.ScopeMatches (deployPlain Readable.relation) ∧
      Claims.Flock.KS H (deployPlain Readable.relation) ∧
      Claims.Flock.Bits H (deployPlain Readable.relation) ∧
      (∀ a b : Claims.Msg → ℝ, (∀ x, Claims.Flock.κ (deployPlain Readable.relation) x ≤ 2 ^ (-a x)) →
        (∀ x, Claims.Flock.γ (deployPlain Readable.relation) x ≤ 2 ^ b x) →
        Claims.Bits H (deployPlain Readable.relation).V (Claims.Flock.R (deployPlain Readable.relation).parse)
          (Claims.Flock.E (deployPlain Readable.relation) Claims.GS.plainDecodeRD)
          fun x => min (a x) (255.5 - b x / 2) - 1) ∧
      Claims.Flock.Efficient H (deployPlain Readable.relation) plainStepPoly ∧
      Claims.Flock.Complete H (deployPlain Readable.relation) :=
  flock_plain_assembled Readable.relation H hKS (FlockComplete.Replay.complete_plain_json H S)

/-- **Flock's claims at the deployed masked (`--zk`) verifier**, at every relation's fields and every hash: the
readable scope is the verifier's; KS at `E^rep`; bits at every decoder meeting `ListWithin`, and at the radius decoder
at every bound on `κ` and `γ`; efficiency at the radius decoder with its count; and completeness at the record's
reference prover (`FlockComplete.ZkTables.honestP` at `FlockComplete.ZkTables.δc`). `hKS` (E4) and `hComplete` are what
remains. -/
theorem flock_zk_assembled (rel : Readable.Parsed → Readable.Relation) (H : Claims.Msg → Claims.Msg)
    (hKS : Claims.Flock.KS H (deployZk rel))
    (hComplete : Claims.Flock.Complete H (deployZk rel)) :
    Claims.Flock.ScopeMatches (deployZk rel) ∧
      Claims.Flock.KS H (deployZk rel) ∧
      Claims.Flock.Bits H (deployZk rel) ∧
      (∀ a b : Claims.Msg → ℝ, (∀ x, Claims.Flock.κ (deployZk rel) x ≤ 2 ^ (-a x)) →
        (∀ x, Claims.Flock.γ (deployZk rel) x ≤ 2 ^ b x) →
        Claims.Bits H (deployZk rel).V (Claims.Flock.R (deployZk rel).parse)
          (Claims.Flock.E (deployZk rel) Claims.GS.zkDecodeRD) fun x => min (a x) (255.5 - b x / 2) - 1) ∧
      Claims.Flock.Efficient H (deployZk rel) zkStepPoly ∧
      Claims.Flock.Complete H (deployZk rel) :=
  ⟨scopeMatches_zk rel, hKS, Claims.Flock.Bits_of_KS hKS (εtable_nonneg true),
    fun _ _ hκ hγ => Claims.Flock.Bits_ge hKS (εtable_nonneg true) hκ
      (fun x => γ_pos (deployZk rel) x (V_k_pos true x)) hγ _ (listWithin_zk rel),
    efficient_zk H rel, hComplete⟩

/-- **Flock's claims at the deployed masked verifier at the deployed relation** (`Readable.relation`), at every hash:
`flock_zk_assembled` with its completeness from the record's JSON and the honest masked run's premises
(`FlockComplete.ZkTables.complete_json`). `hKS` (E4) and those premises are what remains. -/
theorem flock_zk_assembled_relation (H : Claims.Msg → Claims.Msg)
    (hKS : Claims.Flock.KS H (deployZk Readable.relation))
    (S : Claims.JsonSpec FlockComplete.printJ Flock.canon)
    (hPR : ∀ c : FlockComplete.ZkTables.Case H Readable.relation, FlockComplete.ZkTables.ProverRunsAt c)
    (hR : ∀ c : FlockComplete.ZkTables.Case H Readable.relation, FlockComplete.ZkTables.RepShapedAt c)
    (hC : ∀ o : FlockComplete.ZkTables.Run H Readable.relation,
      FlockComplete.ZkTables.LgInnerAt H Readable.relation o) :
    Claims.Flock.ScopeMatches (deployZk Readable.relation) ∧
      Claims.Flock.KS H (deployZk Readable.relation) ∧
      Claims.Flock.Bits H (deployZk Readable.relation) ∧
      (∀ a b : Claims.Msg → ℝ, (∀ x, Claims.Flock.κ (deployZk Readable.relation) x ≤ 2 ^ (-a x)) →
        (∀ x, Claims.Flock.γ (deployZk Readable.relation) x ≤ 2 ^ b x) →
        Claims.Bits H (deployZk Readable.relation).V (Claims.Flock.R (deployZk Readable.relation).parse)
          (Claims.Flock.E (deployZk Readable.relation) Claims.GS.zkDecodeRD)
          fun x => min (a x) (255.5 - b x / 2) - 1) ∧
      Claims.Flock.Efficient H (deployZk Readable.relation) zkStepPoly ∧
      Claims.Flock.Complete H (deployZk Readable.relation) :=
  flock_zk_assembled Readable.relation H hKS (FlockComplete.ZkTables.complete_json H S hPR hR hC)

end FlockDeploy.Assembled
