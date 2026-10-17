import Proofs.Flock.Claims.DeployEfficient
import Proofs.Flock.Claims.Sampling.Leaf
import Proofs.Flock.Claims.Sampling.Runs
import Proofs.Flock.Claims.Sampling.Session
import Proofs.Flock.SamplingBoundary

/-!
# The sampling block's claims, with what remains as hypotheses

`sampling_assembled`: at a sampling deployment `D` whose sessions are Flock's deployed masked verifier
(`D.flock = FlockDeploy.deployZk rel`), at every hash `H`, the conjunction of the sampling claims:

* `ScopeMatches`, `Universal`, `Complete` and `ZK`: hypotheses;
* `Binding`, at hm96 leaves under the registered frame-v3-sha512 path check (`Binding_of_hm96`);
* `FaultyBoundary` (`faultyBoundary`);
* `KS`, on the statements in scope (`Scope`), from Flock's KS at the checked units' sessions (`SessionKS_of`, `KS_of`),
  with the binding;
* `KSCount` from `KS` (`KSCount_of_KS`);
* `Bits` from `KS` and the extractor's expected reruns and own calls (`Bits_of_KS`);
* `Efficient` at `FlockDeploy.zkStepPoly`: the extractor's reruns (`E_runs_le`) and own calls (`E_own_le`) at the
  record's decoder, its expected reruns, and that decoder's `ListWithin` and step count at `deployZk`
  (`FlockDeploy.efficient_zk`).

## What remains: the hypotheses

* `hKS : Claims.Flock.KS H (deployZk rel)`: E4 (`cursor/claims-ks-c7a9`), the hypothesis `flock_zk_assembled` takes.
* `hflock : D.flock = deployZk rel`, `hε : ∀ x, 0 ≤ D.εtable x`: the deployed sampling record's, which is the Lean
  sampled-proofs verifier's (`cursor/sampled-lean-verifier-c7a9`): its sessions are C-Flock's masked proofs, and its
  inner `ε_table`, which bounds each session's (`InnerNumbers`), a sum of tables' errors as Flock's is.
* `hcommit : D.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer` and `hrow : RowPlaces row`: the
  deployed trace commitment's opening program, hm96 leaves of rows that place the opened bit under the verifier's own
  registered path check; that verifier's port of the opening.
* `DrawFixed`, `SessionRuns`, `InnerNumbers` (`Sampling.Session`): the checked units' sessions are in the second stage
  and the draw doesn't read them; a checked unit's session is a Flock run on its own rounds that decides the run, with
  the Flock verifier's `H` calls among the sampled verifier's; the inner numbers are Flock's at each session. The Lean
  sampled-proofs verifier, held to the Python by difftest.
* `DrawLaw`, `SessionOpens`, `InteriorFixed`, `Refuses` (`Sampling.KS`): the same verifier, starting with the draw.
* `hrunsExp`: `Efficient`'s expected reruns on `E`, from the rerun programs' counts (`Sampling.Cost`: `accExp_le`);
  not yet assembled.
* `hopen : OpenCallsLe H D`: checking what one checked unit's Flock witness reveals makes at most `openCalls` SHA-512
  calls; the deployed opening check's count, the Lean sampled-proofs verifier's. With it the extractor's own calls are
  `Efficient`'s (`E_own_le`); its reruns are, at every deployment (`E_runs_le`).
* `hscope : ScopeMatches D`: the deployed scope check, the Lean sampled-proofs verifier's.
* `huniv : Universal D`: the universal unit's Definition (`verity/ml/private_circuits`) against `Netlist.eval`; no
  branch proves it yet.
* `hComplete : Complete H D P δc`: waits on the reference prover (ruling 4), so `P` and `δc` are any.
* `hZK : ZK H D Pz S εzk γzk`: the masked sessions' zero knowledge with the leaf layer's simulator, its binding term
  `γzk` times `Fork`'s collisions among the honest prover's calls in two real runs on one nonce (#2201); no branch
  proves it yet, so `Pz`, `S`, `εzk` and `γzk` are any.
-/

namespace Claims.Sampling

set_option synthInstance.maxSize 1024 in
/-- **The sampling claims at a deployment whose sessions are Flock's masked verifier**, at every hash: the readable
scope is the verifier's; the trace commitment binds; the universal unit computes its netlist; the faulty set is the
boundary form's; KS, its count form and bits on the statements in scope at every decoder meeting Flock's `ListWithin`;
efficiency at the record's decoder; completeness; and zero knowledge. The hypotheses are what remains (the module
docstring says what discharges each). -/
theorem sampling_assembled (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (D : Deploy FlockDeploy.Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace) (H : Msg → Msg)
    (hflock : D.flock = FlockDeploy.deployZk rel) (hKS : Claims.Flock.KS H (FlockDeploy.deployZk rel))
    (hε : ∀ x, 0 ≤ D.εtable x)
    (own : Array _root_.Flock.Registered.Port) (at_ : List Msg → ℕ → Option (Msg × ℕ))
    (row salt π : Opening Bool → Msg) (layer : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg)
    (hcommit : D.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer) (hrow : RowPlaces row)
    (hfixed : DrawFixed D) (hsessions : SessionRuns H D) (hnum : InnerNumbers D) (hdraw : DrawLaw D)
    (iroots : Msg → Seen → ℕ → List Msg) (hopens : SessionOpens H D iroots) (hfix : InteriorFixed H D iroots)
    (hrefuse : Refuses H D)
    (hrunsExp : ∀ decode, ExtractRunsExpLe H (E D decode)
      (fun x => 1 + D.k x + (D.k x : ℝ) ^ 2 * (1 + nPU D x * D.innerWaits x * D.innerK x)))
    (hopen : OpenCallsLe H D)
    (hscope : ScopeMatches D) (huniv : Universal D)
    {Ω : Type} (P : Wit → Prover Ω) (δc : Msg → ℝ) (hComplete : Complete H D P δc)
    {Nz Ωz : Type} [Fintype Nz] [Nonempty Nz] [Fintype Ωz] [Nonempty Ωz] (Pz : Wit → Prover (Nz × Ωz)) (S : SimLeak)
    (εzk γzk : Msg → ℝ) (hZK : ZK H D Pz S εzk γzk) :
    ScopeMatches D ∧ Binding D ∧ Universal D ∧ FaultyBoundary ∧ KS H D ∧ KSCount H D ∧ Bits H D ∧
      Efficient H D FlockDeploy.zkStepPoly ∧ Complete H D P δc ∧ ZK H D Pz S εzk γzk := by
  have hbind : Binding D := Binding_of_hm96 D own at_ row salt π layer hcommit hrow
  have hKS' : Claims.Flock.KS H D.flock := by
    rw [hflock]
    exact hKS
  have hks : KS H D := KS_of iroots (hbind H) (SessionKS_of hKS' hfixed hsessions hnum hε) hopens hfix hdraw hrefuse hε
  have hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ D.flock.decode ∧
      ∀ os, (D.flock.dec os).2 ≤ FlockDeploy.zkStepPoly.eval (Claims.Flock.wordLen os) := by
    rw [hflock]
    exact ⟨(FlockDeploy.efficient_zk H rel).2.1, (FlockDeploy.efficient_zk H rel).2.2.1⟩
  exact ⟨hscope, hbind, huniv, faultyBoundary, hks, KSCount_of_KS hks, Bits_of_KS hks hrunsExp (E_own_le hopen) hε,
    ⟨E_runs_le H D _, hrunsExp _, E_own_le hopen _, hdec.1, hdec.2⟩, hComplete, hZK⟩

end Claims.Sampling
