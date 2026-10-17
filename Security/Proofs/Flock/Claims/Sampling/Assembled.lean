import Proofs.Flock.Claims.GS.RadiusDeploy
import Proofs.Flock.Claims.Sampling.Leaf
import Proofs.Flock.Claims.Sampling.Select.Scope
import Proofs.Flock.Claims.Sampling.Session
import Proofs.Flock.SamplingBoundary

/-!
# The sampling block's claims, with what remains as hypotheses

`sampling_assembled`: at a sampling deployment `D` whose sessions are Flock's deployed masked verifier
(`D.flock = FlockDeploy.deployZk rel`), at every hash `H`, the conjunction of the sampling claims:

* `ScopeMatches`, from the deployed design's well-formedness (`scopeMatches_of`, `designWellFormed`) where the scope
  check is the readable scope's program part;
* `Universal`, `Complete` and `ZK`: hypotheses;
* `Binding`, at hm96 leaves under the registered frame-v3-sha512 path check (`Binding_of_hm96`);
* `FaultyBoundary` (`faultyBoundary`);
* `KS` from Flock's KS at the checked units' sessions (`SessionKS_of`, `KS_of`), with the binding and the draw law
  (`drawLaw_of_slots`), and the verifier's refusal of what is out of scope (`refuses_of`);
* `KSCount` from `KS` (`KSCount_of_KS`);
* `Bits` from `KS` and the extractor's expected reruns and own calls (`Bits_of_KS`);
* `Efficient` at Flock's masked radius decoder (`GS.zkDecodeRD`), its `ListWithin` and `DecoderComplexity` at
  `deployZk` (`GS.radius_deployZk`), with the extractor's run bounds.

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
* `DrawSlots` (`Sampling.Select.DrawLaw`): on every statement that parses and whose query evaluates, the partition and
  design are the verifier's own layout and the design `Slots.select` draws, from uniform bytes of the coins, every slot
  checked where it is refused; the Lean sampled-proofs verifier's run, which builds `D`.
* `SessionOpens`, `InteriorFixed` (`Sampling.KS`): the same verifier.
* `ProgramScope`, `ChecksScope` (`Sampling.Select.Scope`): the scope check is the readable scope's program part (a
  topologically ordered program, partitioned into proof units), and an accepted run passed it. The design part is a
  theorem (`designWellFormed`); the program part waits on the claims' program and the verifier's being one, and on the
  claims' proof units covering a `Q_word` unit, whose gates another unit's may interleave.
* `hrunsLe`, `hrunsExp`, `hown`: `Efficient`'s run bounds on `E`, from the rerun programs' counts (`Sampling.Cost`:
  `waitAccC_runs_eq`, `accExp_le`, `ownLog_flatMapRP`); not yet assembled.
* `huniv : Universal D`: the universal unit's Definition (`verity/ml/private_circuits`) against `Netlist.eval`; no
  branch proves it yet.
* `hComplete : Complete H D P δc`: waits on the reference prover (ruling 4), so `P` and `δc` are any.
* `hZK : ZK H D Pz S εzk`: the masked sessions' zero knowledge with the leaf layer's simulator; no branch proves it
  yet, so `Pz`, `S` and `εzk` are any.
-/

namespace Claims.Sampling

set_option synthInstance.maxSize 1024 in
/-- **The sampling claims at a deployment whose sessions are Flock's masked verifier**, at every hash: the readable
scope is the verifier's; the trace commitment binds; the universal unit computes its netlist; the faulty set is the
boundary form's; KS, its count form and bits at every decoder meeting Flock's `ListWithin`; efficiency at Flock's
masked radius decoder; completeness; and zero knowledge. The hypotheses are what remains (the module docstring says
what discharges each). -/
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
    (hrunsLe : ∀ decode, ExtractRunsLe H (E D decode)
      (fun x => 1 + D.cap x + D.k x * (D.cap x + D.k x * (nPU D x * (D.innerWaits x * D.innerCap x)))))
    (hrunsExp : ∀ decode, ExtractRunsExpLe H (E D decode)
      (fun x => 1 + D.k x + (D.k x : ℝ) ^ 2 * (1 + nPU D x * D.innerWaits x * D.innerK x)))
    (hown : ∀ decode, ExtractOwnLe H (E D decode) (fun x => D.k x ^ 2 * nPU D x * D.openCalls x))
    (huniv : Universal D)
    {Ω : Type} (P : Wit → Prover Ω) (δc : Msg → ℝ) (hComplete : Complete H D P δc)
    {Ωz : Type} [Fintype Ωz] (Pz : Wit → Prover Ωz) (S : SimLeak) (εzk : Msg → ℝ) (hZK : ZK H D Pz S εzk) :
    ScopeMatches D ∧ Binding D ∧ Universal D ∧ FaultyBoundary ∧ KS H D ∧ KSCount H D ∧ Bits H D ∧
      Efficient H D Claims.GS.zkDecodeRD Claims.GS.zkCountedRD Claims.GS.zkSizeRD
        (Polynomial.C 11 * (Claims.GS.zkPolyR + Polynomial.X)) ∧
      Complete H D P δc ∧ ZK H D Pz S εzk := by
  have hbind : Binding D := Binding_of_hm96 D own at_ row salt π layer hcommit hrow
  have hscope : ScopeMatches D := scopeMatches_of D hslots hprog
  have hrefuse : Refuses H D := refuses_of H D hscope hchecks
  have hKS' : Claims.Flock.KS H D.flock := by
    rw [hflock]
    exact hKS
  have hks : KS H D := KS_of iroots (hbind H) (SessionKS_of hKS' hfixed hsessions hnum hε) hopens hfix
    (drawLaw_of_slots D hslots) hrefuse hε
  have hlist : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ Claims.GS.zkDecodeRD := by
    rw [hflock]
    exact (Claims.GS.radius_deployZk rel).1
  exact ⟨hscope, hbind, huniv, faultyBoundary, hks, KSCount_of_KS hks, Bits_of_KS hks hrunsExp hown hε,
    ⟨hrunsLe _, hrunsExp _, hown _, hlist, (Claims.GS.radius_deployZk rel).2⟩, hComplete, hZK⟩

end Claims.Sampling
