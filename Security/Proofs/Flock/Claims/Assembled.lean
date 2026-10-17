import Proofs.Flock.Claims.Bits
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.DeployEfficient
import Proofs.Flock.Claims.ZK.Hypotheses
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

/-- **Bits at the deployed plain record, from KS alone** (`Flock.Bits_of_KS`): the record's table error is
nonnegative (`εtable_nonneg`). -/
theorem bits_plain (rel : Readable.Parsed → Readable.Relation) (H : Claims.Msg → Claims.Msg)
    (hKS : Claims.Flock.KS H (deployPlain rel)) : Claims.Flock.Bits H (deployPlain rel) :=
  Claims.Flock.Bits_of_KS hKS (εtable_nonneg false)

/-- **Bits at the deployed masked record, from KS alone** (`Flock.Bits_of_KS`), as at the plain one. -/
theorem bits_zk (rel : Readable.Parsed → Readable.Relation) (H : Claims.Msg → Claims.Msg)
    (hKS : Claims.Flock.KS H (deployZk rel)) : Claims.Flock.Bits H (deployZk rel) :=
  Claims.Flock.Bits_of_KS hKS (εtable_nonneg true)

/-- **Bits at bounds at the deployed plain record, from KS alone** (`Flock.Bits_ge`): at the record's own decoder,
`min(a, 255.5 − b/2) − 1` bits at every bound `κ ≤ 2^{−a}`, `γ ≤ 2^b`; `γ > 0` because the verifier takes a round. -/
theorem bits_ge_plain (rel : Readable.Parsed → Readable.Relation) (H : Claims.Msg → Claims.Msg)
    (hKS : Claims.Flock.KS H (deployPlain rel)) (a b : Claims.Msg → ℝ)
    (hκ : ∀ x, Claims.Flock.κ (deployPlain rel) x ≤ 2 ^ (-a x)) (hγ : ∀ x, Claims.Flock.γ (deployPlain rel) x ≤ 2 ^ b x) :
    Claims.Bits H (deployPlain rel).V (Claims.Flock.R (deployPlain rel).parse)
      (Claims.Flock.E (deployPlain rel) (deployPlain rel).decode) fun x => min (a x) (255.5 - b x / 2) - 1 :=
  Claims.Flock.Bits_ge hKS (εtable_nonneg false) hκ (fun x => γ_pos (deployPlain rel) x (V_k_pos false x)) hγ _
    (decode_plain rel ▸ listWithin_plain rel)

/-- **Bits at bounds at the deployed masked record, from KS alone**, as at the plain one. -/
theorem bits_ge_zk (rel : Readable.Parsed → Readable.Relation) (H : Claims.Msg → Claims.Msg)
    (hKS : Claims.Flock.KS H (deployZk rel)) (a b : Claims.Msg → ℝ)
    (hκ : ∀ x, Claims.Flock.κ (deployZk rel) x ≤ 2 ^ (-a x)) (hγ : ∀ x, Claims.Flock.γ (deployZk rel) x ≤ 2 ^ b x) :
    Claims.Bits H (deployZk rel).V (Claims.Flock.R (deployZk rel).parse)
      (Claims.Flock.E (deployZk rel) (deployZk rel).decode) fun x => min (a x) (255.5 - b x / 2) - 1 :=
  Claims.Flock.Bits_ge hKS (εtable_nonneg true) hκ (fun x => γ_pos (deployZk rel) x (V_k_pos true x)) hγ _
    (decode_zk rel ▸ listWithin_zk rel)

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
          (Claims.Flock.E (deployPlain rel) (deployPlain rel).decode) fun x => min (a x) (255.5 - b x / 2) - 1) ∧
      Claims.Flock.Efficient H (deployPlain rel) plainStepPoly ∧
      Claims.Flock.Complete H (deployPlain rel) P δc :=
  ⟨scopeMatches_plain rel, hKS, bits_plain rel H hKS, bits_ge_plain rel H hKS, efficient_plain H rel, hComplete⟩

/-- **Flock's claims at the deployed masked (`--zk`) verifier**, at every relation's fields and every hash: the
readable scope is the verifier's; KS at `E^rep`; bits at every decoder meeting `ListWithin`, and at the radius decoder
at every bound on `κ` and `γ`; efficiency at the radius decoder with its count; and completeness at the honest prover.
`hKS` (E4) and `hComplete` (no owner yet) are what remains. -/
theorem flock_zk_assembled (rel : Readable.Parsed → Readable.Relation) (H : Claims.Msg → Claims.Msg)
    (hKS : Claims.Flock.KS H (deployZk rel))
    {Ω : Type} (P : Claims.Flock.Wit → Claims.Prover Ω) (δc : Claims.Msg → ℝ)
    (hComplete : Claims.Flock.Complete H (deployZk rel) P δc)
    (G : Claims.ZKMal.Proto Claims.Flock.Wit) (t : ℕ) (ht : 0 < t) (hK : ∀ x, G.K x = (V true).k x - 1)
    {Salt B Cc : Type} [AddCommGroup Salt] [AddCommGroup B] [Fintype Salt] [DecidableEq Salt] [Fintype B]
    [DecidableEq B] (My : Salt → B) (cs : Salt → Cc) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (Mp : ∀ x, G.Nonce → G.PreModel x Salt B Cc) (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.CoinModel x Salt B Cc)
    (nL nB : Claims.Msg → ℕ) (hnL : ∀ x ν, Fintype.card (Mp x ν).L ≤ nL x) (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.MsgLink H (Claims.Flock.R (deployZk rel).parse) My cs Mp Mc) (hS : G.SimLink H Mc)
    (hN : G.NonDegenerate Mc) (hI : G.InnerHolds (Claims.Flock.R (deployZk rel).parse) Mc) (hC : G.CoinCheck) :
    Claims.Flock.ScopeMatches (deployZk rel) ∧
      Claims.Flock.KS H (deployZk rel) ∧
      Claims.Flock.Bits H (deployZk rel) ∧
      (∀ a b : Claims.Msg → ℝ, (∀ x, Claims.Flock.κ (deployZk rel) x ≤ 2 ^ (-a x)) →
        (∀ x, Claims.Flock.γ (deployZk rel) x ≤ 2 ^ b x) →
        Claims.Bits H (deployZk rel).V (Claims.Flock.R (deployZk rel).parse)
          (Claims.Flock.E (deployZk rel) (deployZk rel).decode) fun x => min (a x) (255.5 - b x / 2) - 1) ∧
      Claims.Flock.ZK H (deployZk rel) G.prover (G.simulatorE t (t * 3 ^ t))
        (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3) ^ t) (fun _ => 4 * t * 3 ^ t) ∧
      Claims.Flock.Efficient H (deployZk rel) zkStepPoly ∧
      Claims.Flock.Complete H (deployZk rel) P δc :=
  ⟨scopeMatches_zk rel, hKS, bits_zk rel H hKS, bits_ge_zk rel H hKS,
    Claims.ZKMal.Flock.zk_deployZk_of_models H rel G t ht hK My cs hT1 hδ₁ Mp Mc nL nB hnL hnB hL hS hN hI hC,
    efficient_zk H rel, hComplete⟩

end FlockDeploy.Assembled
