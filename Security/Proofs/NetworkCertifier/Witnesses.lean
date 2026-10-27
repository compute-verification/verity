import Proofs.NetworkCertifier.SampledAuditInstance
import Proofs.NetworkCertifier.Horizon
import Proofs.Flock.Claims.Witnesses.Sampling
import Proofs.Flock.Claims.Witnesses.Meets
import Proofs.Flock.Claims.KS.ZkExtend
import Definitions.NetworkCertifier.Guarantees.SampledCharge
import Proofs.NetworkCertifier.WitnessToys

/-!
# The honest and nontrivial lemmas of the network certifier's acceptance guarantees

Each guarantee that says what an accepted run establishes has two lemmas (lean-proofs' standing rulings, 26 Oct):
`G_honest`, its hypotheses at the honest deployed instance, and `G_nontrivial`, its conclusion false at an instance
whose verifier accepts everything.

* `SampledAuditAccepted`, `SampledLeakAccepted`, `SampledLeakRegistered`. Their premise `Claims.Sampling.KS` holds at
  SHA-512 at a sampling record over the masked Flock record at the deployed relation, under the assembled record's
  hypotheses (`ks_deployed`: `KS_of` at `sampling_assembled_of`'s, Flock's KS `FlockDeploy.KS.zk_KS_untrim`). So do
  the rest: the record's decoder meets the list-decoding spec, a charge `β` bounds every possible faulty set's
  committed bits at every `δ` (`charge_bounds`), and the hm96 commitment binds. Their acceptance conclusion fails at
  the accepting verifier: `SampledAuditAccepted` at `sksD`, where no opening passes, so the one unit is faulty and no
  witness is in `RelS`; the leak bounds at `leakD`, `sksD` at a statement whose program negates its input. There,
  with nothing opened, the trace holds `false` at the negation, which the circuit computes `true` from the ingress the
  trace carries; the one unit is drawn `all`, so no possible faulty set holds it, and no witness is in `RLeak`.
* `HorizonWrongUnitsLe`. Its draw hypotheses hold at the uniform `k`-subset draw (`uniform_draw`), and its soundness
  hypothesis at `ε = 0` when no unit is wrong. At one window of one unit, that unit wrong, an audit that always accepts,
  `ε = 0` and `q = F = 1`, its bound reads `1 ≤ 0`.

The count conjuncts of the leak bounds read no verifier: they hold at every deployment, the accepting one included.
The instances at the accepting verifier, and the facts about them that read no deployed record, are `WitnessToys`'s;
the monitoring period's pairs, in the lock's `witnesses`, are `PeriodWitnesses`'.
-/

namespace NetTiming.Witness

open Claims Claims.Sampling Claims.Witness Claims.Witness.Sampling

local notation "relD" => _root_.Flock.Readable.relation
local notation "shaD" => FlockDeploy.sha512

/-! ## At the deployed record -/

section Honest

set_option synthInstance.maxSize 1024 in
/-- **`Claims.Sampling.KS` at the deployed record**: at SHA-512, at a sampling record over the masked Flock record at
the deployed relation, under the assembled record's hypotheses, with Flock's KS there
(`FlockDeploy.KS.zk_KS_untrim`). -/
theorem ks_deployed
    (D : Claims.Sampling.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace)
    (hflock : D.flock = FlockDeploy.deployZk relD) (hε : ∀ x, 0 ≤ D.εtable x)
    (own : Array _root_.Flock.Registered.Port) (at_ : List Msg → ℕ → Option (Msg × ℕ))
    (row salt π : Opening Bool → Msg) (layer : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg)
    (hcommit : D.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer) (hrow : RowPlaces row)
    (hfixed : DrawFixed D) (hsessions : SessionRuns shaD D) (hnum : InnerNumbers D) (hslots : DrawSlots D)
    (iroots : Msg → Seen → ℕ → List Msg) (hopens : SessionOpens shaD D iroots) (hfix : InteriorFixed shaD D iroots)
    (hprog : ProgramScope D) (hchecks : ChecksScope shaD D) :
    Claims.Sampling.KS shaD D :=
  Claims.Sampling.KS_of iroots (Binding_of_hm96 D own at_ row salt π layer hcommit hrow shaD)
    (SessionKS_of (by rw [hflock]; exact FlockDeploy.KS.zk_KS_untrim) hfixed hsessions hnum hε) hopens hfix
    (drawLaw_of_slots D hslots) (refuses_of shaD D (scopeMatches_of D hslots hprog) hchecks) hε

/-- **A charge bounds every possible faulty set**, at every deployment and `δ`: each statement's partition's units'
committed bits, all of them. -/
theorem charge_bounds {Coin Col T : Type} (D : Claims.Sampling.Deploy Coin Col T) (δ : ℝ) :
    ∃ β : Claims.Sampling.Stmt → ℕ, ∀ s P, D.partition s = some P →
      ∀ F ∈ faultySets s.design P δ, faultyBits s.prog P F ≤ β s := by
  refine ⟨fun s => ((D.partition s).map fun P => faultyBits s.prog P (Finset.range P.rus.length)).getD 0, ?_⟩
  intro s P hP F hF
  show faultyBits s.prog P F ≤
    ((D.partition s).map fun P => faultyBits s.prog P (Finset.range P.rus.length)).getD 0
  rw [hP]
  show faultyBits s.prog P F ≤ faultyBits s.prog P (Finset.range P.rus.length)
  classical
  simp only [faultySets, Finset.mem_filter, Finset.mem_powerset] at hF
  unfold faultyBits
  exact Finset.sum_le_sum_of_subset hF.1

set_option synthInstance.maxSize 1024 in
/-- **`SampledAuditAccepted`'s hypothesis at the deployed record**: `KS` (`ks_deployed`). -/
theorem SampledAuditAccepted_honest
    (D : Claims.Sampling.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace)
    (hflock : D.flock = FlockDeploy.deployZk relD) (hε : ∀ x, 0 ≤ D.εtable x)
    (own : Array _root_.Flock.Registered.Port) (at_ : List Msg → ℕ → Option (Msg × ℕ))
    (row salt π : Opening Bool → Msg) (layer : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg)
    (hcommit : D.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer) (hrow : RowPlaces row)
    (hfixed : DrawFixed D) (hsessions : SessionRuns shaD D) (hnum : InnerNumbers D) (hslots : DrawSlots D)
    (iroots : Msg → Seen → ℕ → List Msg) (hopens : SessionOpens shaD D iroots) (hfix : InteriorFixed shaD D iroots)
    (hprog : ProgramScope D) (hchecks : ChecksScope shaD D) :
    Claims.Sampling.KS shaD D :=
  ks_deployed D hflock hε own at_ row salt π layer hcommit hrow hfixed hsessions hnum hslots iroots hopens hfix hprog
    hchecks

set_option synthInstance.maxSize 1024 in
/-- **`SampledLeakAccepted`'s hypotheses at the deployed record**: `SampledAuditAccepted`, `KS`, the record's decoder
within Flock's radius (`Sampling.decoder_meets_zk`), and at every `δ` a charge bounding every possible faulty set
(`charge_bounds`). -/
theorem SampledLeakAccepted_honest
    (D : Claims.Sampling.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace)
    (hflock : D.flock = FlockDeploy.deployZk relD) (hε : ∀ x, 0 ≤ D.εtable x)
    (own : Array _root_.Flock.Registered.Port) (at_ : List Msg → ℕ → Option (Msg × ℕ))
    (row salt π : Opening Bool → Msg) (layer : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg)
    (hcommit : D.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer) (hrow : RowPlaces row)
    (hfixed : DrawFixed D) (hsessions : SessionRuns shaD D) (hnum : InnerNumbers D) (hslots : DrawSlots D)
    (iroots : Msg → Seen → ℕ → List Msg) (hopens : SessionOpens shaD D iroots) (hfix : InteriorFixed shaD D iroots)
    (hprog : ProgramScope D) (hchecks : ChecksScope shaD D) :
    NetTiming.Guarantees.SampledAuditAccepted ∧ Claims.Sampling.KS shaD D ∧
      Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ D.flock.decode ∧
      ∀ δ : ℝ, ∃ β : Claims.Sampling.Stmt → ℕ, ∀ s P, D.partition s = some P →
        ∀ F ∈ faultySets s.design P δ, faultyBits s.prog P F ≤ β s :=
  ⟨NetTiming.SecurityProofs.SampledAuditAccepted,
    ks_deployed D hflock hε own at_ row salt π layer hcommit hrow hfixed hsessions hnum hslots iroots hopens hfix hprog
      hchecks,
    Claims.Sampling.decoder_meets_zk relD D hflock, charge_bounds D⟩

set_option synthInstance.maxSize 1024 in
/-- **`SampledLeakRegistered`'s hypotheses at the deployed record**: `SampledLeakAccepted`'s, and the hm96 commitment
binds (`Binding_of_hm96`). -/
theorem SampledLeakRegistered_honest
    (D : Claims.Sampling.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace)
    (hflock : D.flock = FlockDeploy.deployZk relD) (hε : ∀ x, 0 ≤ D.εtable x)
    (own : Array _root_.Flock.Registered.Port) (at_ : List Msg → ℕ → Option (Msg × ℕ))
    (row salt π : Opening Bool → Msg) (layer : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg)
    (hcommit : D.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer) (hrow : RowPlaces row)
    (hfixed : DrawFixed D) (hsessions : SessionRuns shaD D) (hnum : InnerNumbers D) (hslots : DrawSlots D)
    (iroots : Msg → Seen → ℕ → List Msg) (hopens : SessionOpens shaD D iroots) (hfix : InteriorFixed shaD D iroots)
    (hprog : ProgramScope D) (hchecks : ChecksScope shaD D) :
    NetTiming.Guarantees.SampledAuditAccepted ∧ Claims.Sampling.KS shaD D ∧ D.commit.Binds shaD ∧
      Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ D.flock.decode ∧
      ∀ δ : ℝ, ∃ β : Claims.Sampling.Stmt → ℕ, ∀ s P, D.partition s = some P →
        ∀ F ∈ faultySets s.design P δ, faultyBits s.prog P F ≤ β s :=
  ⟨NetTiming.SecurityProofs.SampledAuditAccepted,
    ks_deployed D hflock hε own at_ row salt π layer hcommit hrow hfixed hsessions hnum hslots iroots hopens hfix hprog
      hchecks,
    Binding_of_hm96 D own at_ row salt π layer hcommit hrow shaD,
    Claims.Sampling.decoder_meets_zk relD D hflock, charge_bounds D⟩

end Honest

/-! ## At the accepting verifier -/

section Toy

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **`SampledAuditAccepted`'s conclusion fails at the accepting verifier** `sksD`, at a horizon of no step: no opening
passes, so no witness is in `RelS`, and `RAcc` reads `RelS`. -/
theorem SampledAuditAccepted_nontrivial (H : Msg → Msg) :
    ¬ KSAccepted H (sksD Coin) 1 1 1 (fun _ => trivLay) (fun _ => 0) := by
  intro h
  have h1 := h 0 (fun _ => [()]) le_rfl (Claims.Witness.Flock.listWithin_unit _ _ _)
  have hscope : ScopeAcc (sksD Coin) (fun _ => trivLay) (fun _ => 0) [] :=
    ⟨sksD_scope [], toyStmt, rfl, fun t ht => absurd ht (Nat.not_lt_zero t), fun t ht => absurd ht (Nat.not_lt_zero t),
      fun t ht => absurd ht (Nat.not_lt_zero t)⟩
  exact sksD_bound_fails (Coin := Coin) H (fun _ => [()]) _ _ []
    (fun w ⟨s, P, hs, hP, hrel, _⟩ => sksD_noR (Coin := Coin) H 0 le_rfl [] w ⟨s, P, hs, hP, hrel⟩)
    (by linarith [sksD_κ (Coin := Coin) []]) (h1 [] hscope Unit silentP)

/-- **`SampledLeakAccepted`'s acceptance conclusion fails at the accepting verifier** `leakD`, at one step at `δ = 0`,
with no input registered (`leakD_noRLeak`). -/
theorem SampledLeakAccepted_nontrivial (H : Msg → Msg) :
    ¬ KSLinOn H (leakD Coin).V (ScopeAcc (leakD Coin) (fun _ => notLay) (fun _ => 1))
      (RLeak H (leakD Coin).commit (leakD Coin).parse (leakD Coin).partition 1 1 1 (fun _ => notLay) (fun _ => 1) 0
        (fun s => AsReg s.prog notLay 1 ∅) 1)
      (Claims.Sampling.E (leakD Coin) fun _ => [()]) (Claims.Sampling.κ (leakD Coin) 0) (Claims.Sampling.γ (leakD Coin))
      (fun _ => 1) (Bext H (leakD Coin).V (Claims.Sampling.E (leakD Coin) fun _ => [()])) := fun h =>
  leakD_bound_fails H _ _ _ [] (leakD_noRLeak H _ []) (by linarith [leakD_κ (Coin := Coin) []])
    (h [] (leakD_scopeAcc []) Unit silentP)

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- With no input registered, the registration opens at every statement `leakD` parses. -/
theorem leakD_regOpens (H : Msg → Msg) (x : Msg) :
    ∀ s, (leakD Coin).parse x = some s → RegOpens H (leakD Coin).commit s ∅ [] :=
  fun _ _ => ⟨by simp, by simp⟩

/-- **`SampledLeakRegistered`'s acceptance conclusion fails at the accepting verifier** `leakD`, at one step at
`δ = 0`, with no input registered: at the statement `[]`, in scope (`leakD_scopeAcc`) with its registration opening
(`leakD_regOpens`), its silent prover's bound fails. -/
theorem SampledLeakRegistered_nontrivial (H : Msg → Msg) :
    ¬ PrBad H (leakD Coin).V
        (RLeak H (leakD Coin).commit (leakD Coin).parse (leakD Coin).partition 1 1 1 (fun _ => notLay)
          (fun _ => 1) 0 (fun s _ => AsReg s.prog notLay 1 ∅ (regVal [])) 1)
        (Claims.Sampling.E (leakD Coin) fun _ => [()]) [] silentP ≤
      Claims.Sampling.κ (leakD Coin) 0 [] +
          Real.sqrt (Claims.Sampling.γ (leakD Coin) [] * PrB H (leakD Coin).V silentP []) +
        (Bext H (leakD Coin).V (Claims.Sampling.E (leakD Coin) fun _ => [()]) Unit silentP []).Pr H +
        (Breg H (leakD Coin).V (Claims.Sampling.E (leakD Coin) fun _ => [()]) (leakD Coin).commit
          (leakD Coin).parse (fun _ => []) Unit silentP []).Pr H :=
  leakD_reg_bound_fails H _ _ _ [] (leakD_noRLeak H _ []) (by linarith [leakD_κ (Coin := Coin) []])

end Toy

/-! ## The horizon -/

/-- **`HorizonWrongUnitsLe`'s draw hypotheses at the honest instance**: the uniform `k`-subset draw draws `k` units
and each `k`-subset with share `1 / C(n, k)`. -/
theorem uniform_draw (n k : ℕ) :
    (∀ x : {S : Finset (Fin n) // S.card = k}, x.1.card = k) ∧
      ∀ S : Finset (Fin n), S.card = k →
        (Finset.univ.filter fun x : {S : Finset (Fin n) // S.card = k} => x.1 = S).card * n.choose k =
          Fintype.card {S : Finset (Fin n) // S.card = k} := by
  refine ⟨fun x => x.2, fun S hS => ?_⟩
  have h1 : (Finset.univ.filter fun x : {S : Finset (Fin n) // S.card = k} => x.1 = S) = {⟨S, hS⟩} := by
    ext x
    simp [Subtype.ext_iff]
  rw [h1, Finset.card_singleton, one_mul, Fintype.card_finset_len, Fintype.card_fin]

/-- **`HorizonWrongUnitsLe`'s soundness hypothesis at the honest instance**: at the uniform `k`-subset draw
(`uniform_draw`) with no unit wrong, every acceptance event meets a wrong unit with probability `0`, so `ε = 0`. -/
theorem HorizonWrongUnitsLe_honest (n k : ℕ) :
    ∀ (m : ℕ) (acc : Fin m → (Fin m → {S : Finset (Fin n) // S.card = k}) → Prop) (i : Fin m),
      Pr (fun ω => acc i ω ∧ ¬ Disjoint (ω i).1 (∅ : Finset (Fin n))) ≤ 0 := by
  intro m acc i
  simp [Pr]

/-- **`HorizonWrongUnitsLe`'s conclusion fails at an audit that always accepts**, at `ε = 0`: one window of one unit,
that unit wrong, `q = F = 1`. -/
theorem HorizonWrongUnitsLe_nontrivial :
    ¬ Pr (fun _ : Fin 1 → Unit => (∀ _ : Fin 1, True) ∧
        1 ≤ ∑ _ : Fin 1, (Finset.univ : Finset (Fin 1)).card) ≤
      (1 - 1 : ℝ) ^ 1 + ∑ _ : Fin 1, (0 : ℝ) := by
  simp [Pr]

end NetTiming.Witness
