import Proofs.Flock.Claims.Sampling.Assembled
import Proofs.Flock.Claims.Assembled

/-!
# The sampling block's claims at the plain verifier

`sampling_assembled_plain`: `sampling_assembled` at a sampling deployment whose sessions are Flock's deployed plain
verifier, as the record Flock's claims read (`D.flock = FlockDeploy.deployPlainTrim rel`), with no zero-knowledge
hypothesis or conclusion. Every step of `sampling_assembled` but the decoder's is stated at any record; the decoder's
`ListWithin` and step count are the plain record's (`FlockDeploy.listWithin_plainTrim`, `FlockDeploy.decode_plain`,
`FlockDeploy.efficient_plain`), at `FlockDeploy.plainStepPoly`. `hKS` is Flock's plain KS at that record, the one
`flock_plain_assembled` takes and returns. The other hypotheses are `sampling_assembled`'s, and its module docstring
says what discharges each.

`sampling_assembled_plain_sha512`: at SHA-512 at the deployed relation, with `hKS` from the live sessions' bound with
escapes on accepted runs (`FlockDeploy.KS.plain_KS_of_liveAcc`), as `flock_plain_assembled_trim` takes it.
-/

namespace Claims.Sampling

set_option synthInstance.maxSize 1024 in
/-- **The sampling claims at a deployment whose sessions are Flock's plain verifier**, at every hash: the readable scope
is the verifier's; the trace commitment binds; the universal unit computes its netlist; the faulty set is the boundary
form's; KS, its count form and bits on the statements in scope at every decoder meeting Flock's `ListWithin`;
efficiency at the record's decoder; and completeness. `sampling_assembled` with the plain record's decoder, and no zero knowledge. -/
theorem sampling_assembled_plain (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (D : Deploy FlockDeploy.Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace) (H : Msg → Msg)
    (hflock : D.flock = FlockDeploy.deployPlainTrim rel) (hKS : Claims.Flock.KS H (FlockDeploy.deployPlainTrim rel))
    (hε : ∀ x, 0 ≤ D.εtable x)
    (own : Array _root_.Flock.Registered.Port) (at_ : List Msg → ℕ → Option (Msg × ℕ))
    (row salt π : Opening Bool → Msg) (layer : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg)
    (hcommit : D.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer) (hrow : RowPlaces row)
    (hfixed : DrawFixed D) (hsessions : SessionRuns H D) (hnum : InnerNumbers D) (hslots : DrawSlots D)
    (iroots : Msg → Seen → ℕ → List Msg) (hopens : SessionOpens H D iroots) (hfix : InteriorFixed H D iroots)
    (hprog : ProgramScope D) (hchecks : ChecksScope H D)
    (hopen : OpenCallsLe H D) (huniv : UniversalIs D)
    (hComplete : Complete H D) :
    ScopeMatches D ∧ Binding D ∧ Universal D ∧ FaultyBoundary ∧ KS H D ∧ KSCount H D ∧ Bits H D ∧
      Efficient H D FlockDeploy.plainStepPoly ∧ Complete H D := by
  have hbind : Binding D := Binding_of_hm96 D own at_ row salt π layer hcommit hrow
  have hscope : ScopeMatches D := scopeMatches_of D hslots hprog
  have hrefuse : Refuses H D := refuses_of H D hscope hchecks
  have hKS' : Claims.Flock.KS H D.flock := by
    rw [hflock]
    exact hKS
  have hks : KS H D := KS_of iroots (hbind H) (SessionKS_of hKS' hfixed hsessions hnum hε) hopens hfix
    (drawLaw_of_slots D hslots) hrefuse hε
  have hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ D.flock.decode ∧
      ∀ os, (D.flock.dec os).2 ≤ FlockDeploy.plainStepPoly.eval (Claims.Flock.wordLen os) := by
    rw [hflock, FlockDeploy.decode_plain rel]
    exact ⟨FlockDeploy.listWithin_plainTrim rel, (FlockDeploy.efficient_plain H rel).2.2.1⟩
  exact ⟨hscope, hbind, Universal_of huniv, faultyBoundary, hks, KSCount_of_KS hks,
    Bits_of_KS hks (E_runs_exp H D hfixed) (E_own_le hopen) hε,
    ⟨E_runs_le H D _, E_runs_exp H D hfixed _, E_own_le hopen _, hdec.1, hdec.2⟩, hComplete⟩

open FlockDeploy (sha512) in
set_option synthInstance.maxSize 1024 in
/-- **The sampling claims at the plain verifier, at SHA-512 at the deployed relation**: `sampling_assembled_plain` with
Flock's plain KS from the live sessions' bound with escapes on accepted runs at the statements some run accepts
(`FlockDeploy.KS.plain_KS_of_liveAcc`), the bound `flock_plain_assembled_trim` takes. -/
theorem sampling_assembled_plain_sha512
    (D : Deploy FlockDeploy.Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace)
    (hflock : D.flock = FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation)
    (hl : ∀ x, ¬ Claims.Flock.NoAcc sha512 (FlockDeploy.V false) x → FlockDeploy.KS.PlainLiveAcc x)
    (hε : ∀ x, 0 ≤ D.εtable x)
    (own : Array _root_.Flock.Registered.Port) (at_ : List Msg → ℕ → Option (Msg × ℕ))
    (row salt π : Opening Bool → Msg) (layer : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg)
    (hcommit : D.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer) (hrow : RowPlaces row)
    (hfixed : DrawFixed D) (hsessions : SessionRuns sha512 D) (hnum : InnerNumbers D) (hslots : DrawSlots D)
    (iroots : Msg → Seen → ℕ → List Msg) (hopens : SessionOpens sha512 D iroots) (hfix : InteriorFixed sha512 D iroots)
    (hprog : ProgramScope D) (hchecks : ChecksScope sha512 D)
    (hopen : OpenCallsLe sha512 D) (huniv : UniversalIs D)
    (hComplete : Complete sha512 D) :
    ScopeMatches D ∧ Binding D ∧ Universal D ∧ FaultyBoundary ∧ KS sha512 D ∧ KSCount sha512 D ∧ Bits sha512 D ∧
      Efficient sha512 D FlockDeploy.plainStepPoly ∧ Complete sha512 D :=
  sampling_assembled_plain _root_.Flock.Readable.relation D sha512 hflock (FlockDeploy.KS.plain_KS_of_liveAcc hl) hε
    own at_ row salt π layer hcommit hrow hfixed hsessions hnum hslots iroots hopens hfix hprog hchecks hopen huniv
    hComplete

end Claims.Sampling
