import Proofs.Flock.Claims.Bits
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.GS.RadiusDeploy
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
* `Efficient` at the radius decoder (`GS.plainDecodeRD`, `GS.zkDecodeRD`): `E^rep`'s reruns at every deployment
  (`ERep_cost`), the decoder's `ListWithin` and `DecoderComplexity` (`GS.radius_deployPlain`,
  `GS.radius_deployZk`), and its lists at most `p(size)` long, which bounds `assemble`'s checks
  (`GS.plainDecodeRD_length_le`, `GS.zkDecodeRD_length_le`);
* `Complete` at the honest prover `P` with error `δc`: a hypothesis.

## What remains: the hypotheses

* `hKS : Flock.KS H (deployPlain rel)`, and at `deployZk rel`: E4, `cursor/claims-ks-c7a9`. There KS at SHA-512 rests
  on `TableBoundθ` at the statements that set up (`plain_KSθ_of_setup`, `zk_KSθ_of_setup`; `ConflictTied` and
  `ColsOk` are proved there), but it concludes the branch's radius form `KSθ`, not `Flock.KS`. `KSθ`'s extractor
  decodes the word filled to table 0's level-0 length with a zero column of the lanes' width (`Eθ`, `fillWord`,
  `zeroCol`), at every decoder meeting `ListCompleteθ`; `Flock.KS`'s decodes the opened columns, at every decoder
  meeting `ListWithin`, whose word has `[]` where no column was opened. So the hypothesis is `Flock.KS` itself.
* `hComplete : Flock.Complete H (deployPlain rel) P δc`: C3, `cursor/claims-complete-c7a9`, whose honest prover
  (`honestP`, with its `δc`) is not on main, so `P` and `δc` are any. At `deployZk rel` no branch proves it yet.
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
    (hKS : Claims.Flock.KS H (deployPlain rel))
    {Ω : Type} (P : Claims.Flock.Wit → Claims.Prover Ω) (δc : Claims.Msg → ℝ)
    (hComplete : Claims.Flock.Complete H (deployPlain rel) P δc) :
    Claims.Flock.ScopeMatches (deployPlain rel) ∧
      Claims.Flock.KS H (deployPlain rel) ∧
      Claims.Flock.Bits H (deployPlain rel) ∧
      (∀ a b : Claims.Msg → ℝ, (∀ x, Claims.Flock.κ (deployPlain rel) x ≤ 2 ^ (-a x)) →
        (∀ x, Claims.Flock.γ (deployPlain rel) x ≤ 2 ^ b x) →
        Claims.Bits H (deployPlain rel).V (Claims.Flock.R (deployPlain rel).parse)
          (Claims.Flock.E (deployPlain rel) Claims.GS.plainDecodeRD) fun x => min (a x) (255.5 - b x / 2) - 1) ∧
      Claims.Flock.Efficient H (deployPlain rel) Claims.GS.plainDecodeRD Claims.GS.plainCountedRD
        Claims.GS.plainSizeRD (Polynomial.C 14 * (Claims.GS.flockPolyR + Polynomial.X)) ∧
      Claims.Flock.Complete H (deployPlain rel) P δc :=
  ⟨scopeMatches_plain rel, hKS, Claims.Flock.Bits_of_KS hKS (εtable_nonneg false),
    fun _ _ hκ hγ => Claims.Flock.Bits_ge hKS (εtable_nonneg false) hκ
      (fun x => γ_pos (deployPlain rel) x (V_k_pos false x)) hγ _ (Claims.GS.listWithin_deployPlain rel),
    Claims.Flock.Efficient_of_decoder H _ _ _ _ _ (Claims.GS.radius_deployPlain rel).1
      (Claims.GS.radius_deployPlain rel).2 Claims.GS.plainDecodeRD_length_le,
    hComplete⟩

/-- **Flock's claims at the deployed masked (`--zk`) verifier**, at every relation's fields and every hash: the
readable scope is the verifier's; KS at `E^rep`; bits at every decoder meeting `ListWithin`, and at the radius decoder
at every bound on `κ` and `γ`; efficiency at the radius decoder with its count; and completeness at the honest prover.
`hKS` (E4) and `hComplete` (no owner yet) are what remains. -/
theorem flock_zk_assembled (rel : Readable.Parsed → Readable.Relation) (H : Claims.Msg → Claims.Msg)
    (hKS : Claims.Flock.KS H (deployZk rel))
    {Ω : Type} (P : Claims.Flock.Wit → Claims.Prover Ω) (δc : Claims.Msg → ℝ)
    (hComplete : Claims.Flock.Complete H (deployZk rel) P δc) :
    Claims.Flock.ScopeMatches (deployZk rel) ∧
      Claims.Flock.KS H (deployZk rel) ∧
      Claims.Flock.Bits H (deployZk rel) ∧
      (∀ a b : Claims.Msg → ℝ, (∀ x, Claims.Flock.κ (deployZk rel) x ≤ 2 ^ (-a x)) →
        (∀ x, Claims.Flock.γ (deployZk rel) x ≤ 2 ^ b x) →
        Claims.Bits H (deployZk rel).V (Claims.Flock.R (deployZk rel).parse)
          (Claims.Flock.E (deployZk rel) Claims.GS.zkDecodeRD) fun x => min (a x) (255.5 - b x / 2) - 1) ∧
      Claims.Flock.Efficient H (deployZk rel) Claims.GS.zkDecodeRD Claims.GS.zkCountedRD Claims.GS.zkSizeRD
        (Polynomial.C 11 * (Claims.GS.zkPolyR + Polynomial.X)) ∧
      Claims.Flock.Complete H (deployZk rel) P δc :=
  ⟨scopeMatches_zk rel, hKS, Claims.Flock.Bits_of_KS hKS (εtable_nonneg true),
    fun _ _ hκ hγ => Claims.Flock.Bits_ge hKS (εtable_nonneg true) hκ
      (fun x => γ_pos (deployZk rel) x (V_k_pos true x)) hγ _ (Claims.GS.listWithin_deployZk rel),
    Claims.Flock.Efficient_of_decoder H _ _ _ _ _ (Claims.GS.radius_deployZk rel).1 (Claims.GS.radius_deployZk rel).2
      Claims.GS.zkDecodeRD_length_le,
    hComplete⟩

end FlockDeploy.Assembled
