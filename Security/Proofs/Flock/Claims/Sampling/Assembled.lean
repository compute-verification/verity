import Proofs.Flock.Claims.DeployEfficient
import Proofs.Flock.Claims.Sampling.Leaf
import Proofs.Flock.Claims.Sampling.Select.Scope
import Proofs.Flock.Claims.Sampling.Runs
import Proofs.Flock.Claims.Sampling.RunsExp
import Proofs.Flock.Claims.Sampling.Session
import Proofs.Flock.Claims.Sampling.Universal
import Proofs.Flock.SamplingBoundary

/-!
# The sampling block's claims, with what remains as hypotheses

`sampling_assembled`: at a sampling deployment `D` whose sessions are Flock's deployed masked verifier
(`D.flock = FlockDeploy.deployZk rel`), at every hash `H`, the conjunction of the sampling claims:

* `Complete` and `ZK`: hypotheses;
* `ScopeMatches`, from the deployed design's well-formedness (`scopeMatches_of`, `designWellFormed`) where the scope
  check is the readable scope's program part;
* `Universal`, from the unit's Definition (`Universal_of`);
* `Binding`, at hm96 leaves under the registered frame-v3-sha512 path check (`Binding_of_hm96`);
* `FaultyBoundary` (`faultyBoundary`);
* `KS`, on the statements in scope (`Scope`), from Flock's KS at the checked units' sessions (`SessionKS_of`, `KS_of`),
  with the binding, the draw law (`drawLaw_of_slots`) and the verifier's refusal of what is out of scope
  (`refuses_of`);
* `KSCount` from `KS` (`KSCount_of_KS`);
* `Bits` from `KS`, the extractor's expected reruns (`E_runs_exp`) and its own calls (`Bits_of_KS`);
* `Efficient` at `FlockDeploy.zkStepPoly`: the extractor's reruns (`E_runs_le`), expected reruns (`E_runs_exp`, with
  the draw before the sessions) and own calls (`E_own_le`) at the record's decoder, and that decoder's `ListWithin`
  and step count at `deployZk` (`FlockDeploy.efficient_zk`).

## What remains: the hypotheses

* `hKS : Claims.Flock.KS H (deployZk rel)`: E4 (`cursor/claims-ks-c7a9`), the hypothesis `flock_zk_assembled` takes.
* `hflock : D.flock = deployZk rel`, `hε : ∀ x, 0 ≤ D.εtable x`: facts about the deployed record
  (`SamplingDeploy.deploy_flock`, `SamplingDeploy.deploy_εtable_nonneg`): its sessions are C-Flock's masked proofs,
  and its inner `ε_table`, which bounds each session's (`InnerNumbers`), is the tables' count times their bound.
* `hcommit : D.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer` and `hrow : RowPlaces row`: the
  deployed trace commitment's opening program, hm96 leaves of rows that place the opened bit under the verifier's own
  registered path check; at the deployed record by definition and proved (`SamplingDeploy.deploy_commit`,
  `SamplingDeploy.row_places`).
* `DrawFixed`, `SessionRuns`, `InnerNumbers` (`Sampling.Session`): the checked units' sessions are in the second stage
  and the draw doesn't read them; a checked unit's session is a Flock run on its own rounds that decides the run, with
  the Flock verifier's `H` calls among the sampled verifier's; the inner numbers are Flock's at each session. The Lean
  sampled-proofs verifier, held to the Python by difftest.
* `DrawSlots` (`Sampling.Select.DrawLaw`): on every statement that parses and whose query evaluates, the partition has
  the shape of the verifier's own layout, and the design is the one `Slots.select` draws, from uniform bytes of the
  coins, every slot checked where it is refused; the Lean sampled-proofs verifier's run, which builds `D`.
* `SessionOpens`, `InteriorFixed` (`Sampling.KS`): the same verifier.
* `ProgramScope`, `ChecksScope` (`Sampling.Select.Scope`): the scope check is the readable scope's program part (a
  topologically ordered program, partitioned into proof units), and an accepted run passed it. The design part is a
  theorem (`designWellFormed`); the program part waits on the claims' program and the verifier's being one. The
  verifier's reading of the descriptor is total (C-Flock's verifier has no `partial` def); what remains is the
  flattening of the program it decodes into the claims' gates.
* `hopen : OpenCallsLe H D`: checking what one checked unit's Flock witness reveals makes at most `openCalls` SHA-512
  calls; at the deployed record, from the reveal count (`SamplingDeploy.openCallsLe_deploy`). With it the extractor's
  own calls are `Efficient`'s (`E_own_le`); its reruns are, at every deployment (`E_runs_le`), and its expected reruns
  are with `DrawFixed` (`E_runs_exp`).
* `huniv : UniversalIs D`: the deployed universal unit computes what its Definition's gates do (`universalProg`, the
  port of `verity.ml.private_circuits.universal`'s `build`); the code side's binding of the compiled unit. That those
  gates compute `Netlist.eval` is proved (`universal_correct`); the deployed record's unit is the Definition's
  (`SamplingDeploy.universalIs_deploy`).
* `hComplete : Complete H D`: of the record's reference prover (`D.P`) at its error (`D.δc H`); at the deployed record,
  from Flock's masked completeness and the sampled verifier's and the code's facts (`SamplingDeploy.complete_deploy`).
* `hZK : ZK H (D.mask Nz Ωz tapeOf S) εzk γzk`: the zero knowledge of the record's reference prover on the masked tapes
  `tapeOf (ν, r)`, with the leaf layer's simulator `S`, its binding term `γzk` times `Fork`'s collisions among the
  honest prover's calls in two real runs on one nonce (#2201); `Sampling.ZK` reduces it to a committed-coin protocol's
  models (`SamplingDeploy.zk_deploy_of_models_gap`).
-/

namespace Claims.Sampling

/-- **`D` masked**: the record with its reference prover's tape read off a nonce and the rest (`tapeOf`), and the
simulator `S`. -/
def Deploy.mask {Coin Col T : Type} (D : Deploy Coin Col T) (Nonce Rest : Type) [Fintype Nonce] [Nonempty Nonce]
    [Fintype Rest] [Nonempty Rest] (tapeOf : Nonce × Rest → D.Tape) (S : SimLeak) : ZkDeploy Coin Col T :=
  { toDeploy := D, Nonce := Nonce, Rest := Rest, tapeOf := tapeOf, S := S }

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
    (hfixed : DrawFixed D) (hsessions : SessionRuns H D) (hnum : InnerNumbers D) (hslots : DrawSlots D)
    (iroots : Msg → Seen → ℕ → List Msg) (hopens : SessionOpens H D iroots) (hfix : InteriorFixed H D iroots)
    (hprog : ProgramScope D) (hchecks : ChecksScope H D)
    (hopen : OpenCallsLe H D) (huniv : UniversalIs D)
    (hComplete : Complete H D)
    {Nz Ωz : Type} [Fintype Nz] [Nonempty Nz] [Fintype Ωz] [Nonempty Ωz] (tapeOf : Nz × Ωz → D.Tape) (S : SimLeak)
    (εzk γzk : Msg → ℝ) (hZK : ZK H (D.mask Nz Ωz tapeOf S) εzk γzk) :
    ScopeMatches D ∧ Binding D ∧ Universal D ∧ FaultyBoundary ∧ KS H D ∧ KSCount H D ∧ Bits H D ∧
      Efficient H D FlockDeploy.zkStepPoly ∧ Complete H D ∧ ZK H (D.mask Nz Ωz tapeOf S) εzk γzk := by
  have hbind : Binding D := Binding_of_hm96 D own at_ row salt π layer hcommit hrow
  have hscope : ScopeMatches D := scopeMatches_of D hslots hprog
  have hrefuse : Refuses H D := refuses_of H D hscope hchecks
  have hKS' : Claims.Flock.KS H D.flock := by
    rw [hflock]
    exact hKS
  have hks : KS H D := KS_of iroots (hbind H) (SessionKS_of hKS' hfixed hsessions hnum hε) hopens hfix
    (drawLaw_of_slots D hslots) hrefuse hε
  have hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ D.flock.decode ∧
      ∀ os, (D.flock.dec os).2 ≤ FlockDeploy.zkStepPoly.eval (Claims.Flock.wordLen os) := by
    rw [hflock]
    exact ⟨(FlockDeploy.efficient_zk H rel).2.1, (FlockDeploy.efficient_zk H rel).2.2.1⟩
  exact ⟨hscope, hbind, Universal_of huniv, faultyBoundary, hks, KSCount_of_KS hks,
    Bits_of_KS hks (E_runs_exp H D hfixed) (E_own_le hopen) hε,
    ⟨E_runs_le H D _, E_runs_exp H D hfixed _, E_own_le hopen _, hdec.1, hdec.2⟩, hComplete, hZK⟩

end Claims.Sampling
