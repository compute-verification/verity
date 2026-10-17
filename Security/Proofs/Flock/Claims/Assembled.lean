import Proofs.Flock.Claims.Bits
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.DeployEfficient
import Proofs.Flock.Soundness.CompiledSound

/-!
# Flock's claims at the deployed records, with what remains as hypotheses

`flock_plain_assembled` and `flock_zk_assembled`: at the records Flock's claims read, `FlockDeploy.deployPlainTrim rel`
and `FlockDeploy.deployZkTrim rel` (the deployed instances with their columns trimmed of trailing zeros), at every
relation's fields `rel` and every hash `H`, the conjunction of Flock's claims:

* `ScopeMatches`: the readable scope is the verifier's (`scopeMatches_plain`, `scopeMatches_zk`);
* `KS` at `E^rep` and every decoder meeting `ListWithin`: a hypothesis;
* `Bits` at every such decoder (`Flock.Bits_of_KS`), and at the radius decoder `min(a, 255.5 − b/2) − 1` bits at every
  bound `κ ≤ 2^{−a}`, `γ ≤ 2^b` (`Flock.Bits_ge`), from `KS` with `ε_table ≥ 0` and `γ > 0` (`εtable_nonneg`,
  `γ_pos`);
* `Efficient` at the record's own counted decoder (`GS.plainCountedRD`, `GS.zkCountedRD`): `E^rep`'s reruns at every
  deployment (`ERep_cost`), the decoder's `ListWithin`, and its count and lists at most `plainStepPoly` (`zkStepPoly`)
  at the opened word's encoded length, which bounds `assemble`'s checks (`efficient_plain`, `efficient_zk`);
* `Complete` at the honest prover `P` with error `δc`: a hypothesis.

## What remains: the hypotheses

* `hKS : Flock.KS H (deployPlainTrim rel)`, and at `deployZkTrim rel`: E4, `cursor/claims-ks-c7a9`. At SHA-512 it
  follows from the fixed-table bound over full-width tables at the untrimmed instance (`FlockDeploy.KS.plain_KS_trim`,
  `FlockDeploy.KS.zk_KS_trim`). There KS at SHA-512 rests on `TableBoundθ` at the statements that set up
  (`plain_KSθ_of_setup`, `zk_KSθ_of_setup`; `ConflictTied` and `ColsOk` are proved there), but it concludes the
  branch's radius form `KSθ`, not `Flock.KS`. `KSθ`'s extractor
  decodes the word filled to table 0's level-0 length with a zero column of the lanes' width (`Eθ`, `fillWord`,
  `zeroCol`), at every decoder meeting `ListCompleteθ`; `Flock.KS`'s decodes the opened columns, at every decoder
  meeting `ListWithin`, whose word has `[]` where no column was opened. So the hypothesis is `Flock.KS` itself.
* `hComplete : Flock.Complete H (deployPlainTrim rel) P δc`: C3, `cursor/claims-complete-c7a9`, whose honest prover
  (`honestP`, with its `δc`) is not on main, so `P` and `δc` are any. At `deployZkTrim rel` no branch proves it yet.
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
on `κ` and `γ`; efficiency at the radius decoder with its count; and completeness at the honest prover. `hKS` (E4) and
`hComplete` (C3) are what remains. -/
theorem flock_plain_assembled (rel : Readable.Parsed → Readable.Relation) (H : Claims.Msg → Claims.Msg)
    (hKS : Claims.Flock.KS H (deployPlainTrim rel))
    {Ω : Type} (P : Claims.Flock.Wit → Claims.Prover Ω) (δc : Claims.Msg → ℝ)
    (hComplete : Claims.Flock.Complete H (deployPlainTrim rel) P δc) :
    Claims.Flock.ScopeMatches (deployPlainTrim rel) ∧
      Claims.Flock.KS H (deployPlainTrim rel) ∧
      Claims.Flock.Bits H (deployPlainTrim rel) ∧
      (∀ a b : Claims.Msg → ℝ, (∀ x, Claims.Flock.κ (deployPlainTrim rel) x ≤ 2 ^ (-a x)) →
        (∀ x, Claims.Flock.γ (deployPlainTrim rel) x ≤ 2 ^ b x) →
        Claims.Bits H (deployPlainTrim rel).V (Claims.Flock.R (deployPlainTrim rel).parse)
          (Claims.Flock.E (deployPlainTrim rel) Claims.GS.plainDecodeRD) fun x => min (a x) (255.5 - b x / 2) - 1) ∧
      Claims.Flock.Efficient H (deployPlainTrim rel) plainStepPoly ∧
      Claims.Flock.Complete H (deployPlainTrim rel) P δc :=
  ⟨scopeMatches_plain rel, hKS, Claims.Flock.Bits_of_KS hKS (εtable_nonneg false),
    fun _ _ hκ hγ => Claims.Flock.Bits_ge hKS (εtable_nonneg false) hκ
      (fun x => γ_pos (deployPlainTrim rel) x (V_k_pos false x)) hγ _ (listWithin_plain rel),
    efficient_plain H rel, hComplete⟩

/-- **Flock's claims at the deployed masked (`--zk`) verifier**, at every relation's fields and every hash: the
readable scope is the verifier's; KS at `E^rep`; bits at every decoder meeting `ListWithin`, and at the radius decoder
at every bound on `κ` and `γ`; efficiency at the radius decoder with its count; and completeness at the honest prover.
`hKS` (E4) and `hComplete` (no owner yet) are what remains. -/
theorem flock_zk_assembled (rel : Readable.Parsed → Readable.Relation) (H : Claims.Msg → Claims.Msg)
    (hKS : Claims.Flock.KS H (deployZkTrim rel))
    {Ω : Type} (P : Claims.Flock.Wit → Claims.Prover Ω) (δc : Claims.Msg → ℝ)
    (hComplete : Claims.Flock.Complete H (deployZkTrim rel) P δc) :
    Claims.Flock.ScopeMatches (deployZkTrim rel) ∧
      Claims.Flock.KS H (deployZkTrim rel) ∧
      Claims.Flock.Bits H (deployZkTrim rel) ∧
      (∀ a b : Claims.Msg → ℝ, (∀ x, Claims.Flock.κ (deployZkTrim rel) x ≤ 2 ^ (-a x)) →
        (∀ x, Claims.Flock.γ (deployZkTrim rel) x ≤ 2 ^ b x) →
        Claims.Bits H (deployZkTrim rel).V (Claims.Flock.R (deployZkTrim rel).parse)
          (Claims.Flock.E (deployZkTrim rel) Claims.GS.zkDecodeRD) fun x => min (a x) (255.5 - b x / 2) - 1) ∧
      Claims.Flock.Efficient H (deployZkTrim rel) zkStepPoly ∧
      Claims.Flock.Complete H (deployZkTrim rel) P δc :=
  ⟨scopeMatches_zk rel, hKS, Claims.Flock.Bits_of_KS hKS (εtable_nonneg true),
    fun _ _ hκ hγ => Claims.Flock.Bits_ge hKS (εtable_nonneg true) hκ
      (fun x => γ_pos (deployZkTrim rel) x (V_k_pos true x)) hγ _ (listWithin_zk rel),
    efficient_zk H rel, hComplete⟩

end FlockDeploy.Assembled
