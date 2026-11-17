module

public import Properties.Remote
public import Proofs.Properties.Composition
public import Proofs.Properties.InferenceOnly.DescVerdict
public import Proofs.Properties.InferenceOnly.Theta
public import Proofs.Properties.InferenceOnly.ThetaSteps
public import Proofs.Properties.InferenceOnly
public import Proofs.Properties.InferenceOnly.Agrees
public import Proofs.Slowdown.FitWitness
public import Proofs.Properties.Spot
public import Proofs.InferenceOnly.CompliesCK
public import Proofs.Core.Expand
public import Std.Data.TreeMap.Raw.Lemmas
import all Lean.Data.Json.Basic

/-!
# Remote's acceptance and the inference-only policy, proved

Remote's check is `Core.Composition.holdTies` of its tie over `Core.Composition.checkSame` over its components on the
caller's blobs (`remote_check`), so an acceptance is `checkSame`'s with the tie held (`Properties.Composition.acceptTied`),
and `checkSame`'s is `Core.Composition.check`'s (`checkSame_accept`), which `Properties.Composition.acceptMeans` reduces
to each component accepting its own case, with SPOT's record the policy's (`strays_two`) and every
Commit record the period's windows name the policy's record too (`executionPhases_tied`). The policy's
acceptance is `InferenceOnlyCheck.Theta.evaluate`'s with the record naming the circuit (`policy_accept`), which is
`InferenceOnlyCheck.Desc.evaluate`'s and θ's provenance (`ThetaSteps.evaluate_ok`), whose verdict `DescReports`,
`DescWitnesses` and `ThetaProvenance` read; `InferenceOnly.SecurityProofs.constInputs_isCK` and `constInputs_expand_isCK`
give the `(c, k)`-partitions, of the declaration with its constants registered, which needs no assumption that it is
Boolean or of positive fan-in; where it is, they are partitions of the declaration itself (`policyMet_old_clauses`).
The parse, declaration and division an acceptance
exhibits come from `DescAgrees` (the verdict is the gate-by-gate check's on an expansion) and `Decodes` (that check's
verdicts are about a parse, a layout, a baseline and a division). SPOT's acceptance is one-stage v1's check
at the coins it read (`OneStage.checkAt`), one-stage v2's on the verdicts its own run of the verifier computed
(`stage_zk_registered`), so its record keeps R2 (`stage_registered_at`): SPOT's params decode to no other form that
accepts (the recursive form is refused by name, the two-stage form refuses while its switch is off). It
reads SPOT's acceptance through the caller's check (`VStar.Recursive.Caller.gate`), whose emptiness gives the caller's
facts (`Properties.Spot.CallerHeld`): `CallerCheckSound`, which `stage_held` takes, is the forward half of
`CallerCheckMeans` (`callerCheckSound_of_means`), which `callerCheckMeans` proves from the check's steps, so
`acceptInferenceOnly` rests on no premise of the caller's (`callerCheckSound`).

The policy's existence clause is neither empty nor trivial: `policyMet_honest` exhibits a declaration and a division into
nonempty inference blocks that meet the policy, and `policyMet_nontrivial` one the policy refuses at every `α`. Both
conclusions fail at the cases a check that accepts every input accepts, `ev = []` (`acceptInferenceOnly_nontrivial`,
`acceptComplies_nontrivial`).

The digests' tie and the cases' decoding are proved over three components left abstract (`accepted_three`,
`strays_two`), then read at remote's, so that no subprotocol's decoder or digest is unfolded by `simp`.
-/

@[expose] public section

namespace Properties.Remote.SecurityProofs

open Core Core.Composition Verity
open Lean (Json)
open Flock.Program (Prog)
open InferenceOnlyCheck.Properties (Parses Expands LaysOut Names Divides Meets ProvenanceHeld Reported Witnessed)
open InferenceOnlyCheck.Protocol (instanceOf trustedOf)

private theorem getObjVal?_mkObj {l : List (String × Json)} (hd : l.Pairwise (fun a b => ¬ compare a.1 b.1 = .eq))
    {k : String} {v : Json} (hm : (k, v) ∈ l) : (Json.mkObj l).getObjVal? k = .ok v := by
  simp only [Json.mkObj, Json.getObjVal?, Std.TreeMap.Raw.get?_eq_getElem?]
  rw [Std.TreeMap.Raw.getElem?_ofList_of_mem (Std.ReflCmp.compare_self) hd hm]
  rfl

theorem instance_program (p : InferenceOnlyCheck.Protocol.Params) (e : InferenceOnlyCheck.Protocol.Evidence) :
    (instanceOf p e).getObjVal? "program" = .ok (.str p.program) :=
  getObjVal?_mkObj (by simp) (by simp)

theorem instance_gamma (p : InferenceOnlyCheck.Protocol.Params) (e : InferenceOnlyCheck.Protocol.Evidence) :
    (instanceOf p e).getObjVal? "gamma" = .ok p.gamma :=
  getObjVal?_mkObj (by simp) (by simp)

theorem instance_alpha (p : InferenceOnlyCheck.Protocol.Params) (e : InferenceOnlyCheck.Protocol.Evidence) :
    (instanceOf p e).getObjVal? "alpha" = .ok p.alpha :=
  getObjVal?_mkObj (by simp) (by simp)

/-- Two parses of one instance agree on its program, `Γ` and `α`. -/
theorem parses_same {j : Json} {i i' : InferenceOnlyCheck.Instance} (h : Parses j i) (h' : Parses j i') :
    i.program = i'.program ∧ i.gamma = i'.gamma ∧ i.num = i'.num ∧ i.den = i'.den := by
  obtain ⟨hp, -, -, ⟨v, hv, hg⟩, ⟨a, b, hab, ha, hb⟩, -⟩ := h
  obtain ⟨hp', -, -, ⟨v', hv', hg'⟩, ⟨a', b', hab', ha', hb'⟩, -⟩ := h'
  rw [hp] at hp'
  rw [hv] at hv'
  rw [hab] at hab'
  have e2 : v = v' := Except.ok.inj hv'
  have e3 : a = a' ∧ b = b' := by simpa using Json.arr.inj (Except.ok.inj hab')
  obtain ⟨rfl, rfl⟩ := e3
  subst e2
  exact ⟨Json.str.inj (Except.ok.inj hp'), Option.some.inj (hg.symm.trans hg'),
    Option.some.inj (ha.symm.trans ha'), Option.some.inj (hb.symm.trans hb')⟩

/-- Every parse of an expansion of the policy's instance carries the policy's parameters. -/
theorem verifier_own {p : InferenceOnlyCheck.Protocol.Params} {e : InferenceOnlyCheck.Protocol.Evidence} {j' : Json}
    {i : InferenceOnlyCheck.Instance} (hx : Expands (instanceOf p e) j') (hi : Parses j' i) : VerifierOwn p i := by
  obtain ⟨i0, -, h0, -, theta', owner', h0', -, -⟩ := hx
  obtain ⟨hpr, hg, hn, hd⟩ := parses_same h0' hi
  obtain ⟨hp0, -, -, ⟨v, hv, hgv⟩, ⟨a, b, hab, ha, hb⟩, -⟩ := h0
  rw [instance_program] at hp0
  rw [instance_gamma] at hv
  rw [instance_alpha] at hab
  have e1 : p.program = i0.program := Json.str.inj (Except.ok.inj hp0)
  have e2 : p.gamma = v := Except.ok.inj hv
  have e3 : p.alpha = .arr #[a, b] := Except.ok.inj hab
  have hpr' : i0.program = i.program := hpr
  have hg' : i0.gamma = i.gamma := hg
  have hn' : i0.num = i.num := hn
  have hd' : i0.den = i.den := hd
  refine ⟨(e1.trans hpr').symm, ?_, a, b, e3, ?_, ?_⟩
  · rw [e2, hgv, hg']
  · rw [ha, hn']
  · rw [hb, hd']

/-- An expansion of the policy's instance, and what it parses to, carry the registered circuit's text. -/
theorem expands_program {p : InferenceOnlyCheck.Protocol.Params} {e : InferenceOnlyCheck.Protocol.Evidence} {j' : Json}
    {i : InferenceOnlyCheck.Instance} (hx : Expands (instanceOf p e) j') (hi : Parses j' i) : i.program = p.program := by
  obtain ⟨i0, -, h0, -, theta', owner', h0', -, -⟩ := hx
  have a : (instanceOf p e).getObjVal? "program" = .ok (.str i0.program) := h0.1
  rw [instance_program] at a
  have b : j'.getObjVal? "program" = .ok (.str i0.program) := h0'.1
  rw [hi.1] at b
  exact (Json.str.inj (Except.ok.inj b)).trans (Json.str.inj (Except.ok.inj a)).symm

theorem bytes_eq_of_beq {a b : ByteArray} (h : (a == b) = true) : a = b :=
  ByteArray.ext (eq_of_beq (show ByteArray.beq a b = true from h))

theorem bytes_bne_self (a : ByteArray) : (a != a) = false := by
  have h : ByteArray.beq a a = true := by unfold ByteArray.beq; simp
  simp [bne, show (a == a) = true from h]

theorem checkSame_accept {id : String} {ss : List String} {cs : List Protocol} {ps : List Named} {c : ByteArray}
    {ev : List Case} {pub prof : Keyed} (h : checkSame id ss cs ps c ev = .accept pub prof) :
    check id cs ps c ev = .accept pub prof ∧ strays ss cs ev = none := by
  unfold checkSame at h
  unfold check
  rcases hr : run cs ps ev with r | answers <;> simp only [hr] at h ⊢
  · cases h
  · rcases hs : strays ss cs ev with _ | r <;> simp only [hs] at h
    · exact ⟨h, rfl⟩
    · cases h

/-- When `strays` finds no stray, every component's digest is the first's. -/
theorem strays_none {ss : List String} {cs : List Protocol} {ev : List Case} {s : String} {d : ByteArray}
    {rest : List (String × ByteArray)}
    (hm : ss.mapM (fun s => (committedBy s cs ev).map (s, ·)) = some ((s, d) :: rest))
    (h : strays ss cs ev = none) : ∀ x ∈ rest, x.2 = d := by
  unfold strays at h
  simp only [hm] at h
  intro x hx
  cases hf : rest.find? (fun y => y.2 != d) with
  | none =>
    have := List.find?_eq_none.1 hf x hx
    exact bytes_eq_of_beq (by simpa [bne] using this)
  | some y => simp [hf] at h

/-- Remote's check on the caller's blobs is its tie held over `checkSame` over its components on them. -/
theorem remote_check (blobs : Array ByteArray) (ps : List Named) (c : ByteArray) (ev : List Case) :
    (Core.Remote.protocol.on blobs).check ps c ev =
      holdTies [Core.Remote.executionPhases] ev (checkSame "verity/remote/v0"
        ["verity/inference-only/v0", "verity/sampled-proofs/v1"]
        [InferenceOnlyCheck.Protocol.protocol, PouwVerifier.Period.protocol,
          VStar.Recursive.Sampled.protocol.onBlobs (Blobs.ofArray blobs)] ps c ev) := rfl

theorem id_policy : InferenceOnlyCheck.Protocol.protocol.id = "verity/inference-only/v0" := rfl
theorem id_monitoringPeriod : PouwVerifier.Period.protocol.id = "verity/pouw/period/v1" := rfl
theorem id_sampled (bs : Blobs) : (VStar.Recursive.Sampled.protocol.onBlobs bs).id = "verity/sampled-proofs/v1" := rfl

/-- SPOT's commitment and digest are the one-stage registration record's, on any blobs. -/
theorem digestOf_sampled (bs : Blobs) (b : ByteArray) :
    Properties.Composition.digestOf (VStar.Recursive.Sampled.protocol.onBlobs bs) b =
      Properties.Composition.digestOf Spot.OneStage.protocol b := rfl

/-- What the policy's check accepts: a record naming the registered circuit, and a verdict of compliance of the check of
record. -/
theorem policy_accept {p : InferenceOnlyCheck.Protocol.Params} {c : InferenceOnlyCheck.Protocol.Commitment}
    {e : InferenceOnlyCheck.Protocol.Evidence} {pub : Unit} {prof : Json}
    (h : InferenceOnlyCheck.Protocol.check p c e = .accept pub prof) :
    InferenceOnlyCheck.Protocol.names p c = true ∧ ∃ out, InferenceOnlyCheck.Theta.evaluate (instanceOf p e) (trustedOf p c) = .ok out ∧
      out.getObjVal? "complies" = .ok (.bool true) := by
  unfold InferenceOnlyCheck.Protocol.check at h
  by_cases hn : InferenceOnlyCheck.Protocol.names p c = true
  · simp only [hn, Bool.not_true, Bool.false_eq_true, ite_false] at h
    refine ⟨hn, ?_⟩
    split at h
    · cases h
    · rename_i out hout
      split at h
      · rename_i hc
        exact ⟨out, hout, hc⟩
      · cases h
  · simp only [Bool.not_eq_true] at hn
    simp [hn] at h

/-- The record names the circuit the program decodes to. -/
theorem names_program {p : InferenceOnlyCheck.Protocol.Params} {c : InferenceOnlyCheck.Protocol.Commitment} {P : Prog}
    (hn : InferenceOnlyCheck.Protocol.names p c = true) (hP : Flock.Program.decode p.program.toUTF8 = .ok P) :
    InferenceOnlyCheck.Theta.strAt c.record "program" = some (Flock.hexOf P.sha512) := by
  unfold InferenceOnlyCheck.Protocol.names at hn
  rw [hP] at hn
  simpa using hn

/-- `PolicyMet` from its first clause, every inference block meeting both bounds: the rest follows
(`InferenceOnly.CompliesRate`, and the partitions of the declaration with its constants registered by `constInputs_isCK` and
`constInputs_expand_isCK`, on the division with its constants taken out). -/
theorem policyMet_of_meets {d : RegisteredCircuit} {n : ℕ} {U : Partition d.D n} {Γ : ℕ} {α : ℝ}
    (h : ∀ u, Meets d U Γ α u) : PolicyMet d U Γ α :=
  ⟨h, ⟨n, U, h⟩, fun h1 => ⟨n, U.constInputs, InferenceOnly.SecurityProofs.constInputs_isCK h h1⟩,
    fun _ hW => ⟨n, U.constInputs.expand, InferenceOnly.SecurityProofs.constInputs_expand_isCK h hW⟩⟩

/-- **`PolicyMet` keeps its earlier partition clauses**: where their antecedents hold, `d` Boolean, or of positive
fan-in in bits with every computed gate 1 to `W` bits wide, `d` has no constant, so its partitions with its constants
registered are partitions of `d` and of its expansion, with the same units. -/
theorem policyMet_old_clauses {d : RegisteredCircuit} {n : ℕ} {U : Partition d.D n} {Γ : ℕ} {α : ℝ}
    (h : PolicyMet d U Γ α) :
    (d.D.Boolean → ∃ (B : ℕ) (P : Partition d.D B), P.IsCK d Γ ⌊α * Γ⌋₊) ∧
    ∀ W : ℕ, d.D.BitFaninPos → (∀ g, d.D.isInput g = false → 1 ≤ d.D.w g) → (∀ g, d.D.w g ≤ W) →
      ∃ (B : ℕ) (P : Partition d.expand.D B), P.IsCK d.expand (Γ * W) ⌊α * (Γ * W : ℕ)⌋₊ := by
  refine ⟨fun hB => ?_, fun W hf hw hW => ?_⟩
  · have hc := d.D.isConst_false hB.bitFaninPos fun g _ => (hB.one_bit g).ge
    obtain ⟨B, P, hP⟩ := h.2.2.1 hB.one_bit
    refine ⟨B, Partition.ofConstInputs hc P, hB, fun u => hP.card u, fun u => ?_⟩
    rw [← d.constInputs_unregistered_eq hc]
    exact hP.reads u
  · have hc := d.D.isConst_false hf hw
    obtain ⟨B, P, hP⟩ := h.2.2.2 W hW
    refine ⟨B, Partition.ofConstInputsExpand hc P, d.D.expand_boolean hf, fun u => hP.card u, fun u => ?_⟩
    rw [← d.constInputs_expand_unregistered_eq hc]
    exact hP.reads u

/-- **Honest witness of the policy's existence clause**: a declaration and a division into nonempty inference blocks
that meet the policy, `Slowdown.SecurityProofs.FitWitness`'s five-gate circuit in its two one-gate windows, at `Γ = 1` and
`α = 3`. -/
theorem policyMet_honest : ∃ (d : RegisteredCircuit) (n : ℕ) (U : Partition d.D n) (Γ : ℕ) (α : ℝ),
    0 < n ∧ (∀ u, (U.gates u).Nonempty) ∧ PolicyMet d U Γ α := by
  have hs : ∀ u, Slowdown.SecurityProofs.FitWitness.decl.D.size (Slowdown.SecurityProofs.FitWitness.serverBlocks.gates u) = 1 := by
    decide +kernel
  refine ⟨Slowdown.SecurityProofs.FitWitness.decl, 2, Slowdown.SecurityProofs.FitWitness.serverBlocks, 1, 3, by norm_num,
    by decide +kernel, policyMet_of_meets fun u => ⟨(hs u).le, ?_⟩⟩
  rw [hs u, Nat.cast_one, mul_one]
  exact Slowdown.SecurityProofs.FitWitness.reads_le u

/-- **Nontrivial witness of the policy's existence clause**: the policy refuses a declaration and a division into
nonempty inference blocks, the same windows at `Γ = 0`, at every `α`: each block is a gate of width 1. -/
theorem policyMet_nontrivial : ∃ (d : RegisteredCircuit) (n : ℕ) (U : Partition d.D n),
    (∀ u, (U.gates u).Nonempty) ∧ ∀ α : ℝ, ¬ PolicyMet d U 0 α := by
  refine ⟨Slowdown.SecurityProofs.FitWitness.decl, 2, Slowdown.SecurityProofs.FitWitness.serverBlocks, by decide +kernel,
    fun α h => ?_⟩
  have hs : Slowdown.SecurityProofs.FitWitness.decl.D.size (Slowdown.SecurityProofs.FitWitness.serverBlocks.gates 0) = 1 := by
    decide +kernel
  have h0 := (h.1 0).1
  omega

/-- What a verdict of compliance on the policy's instance gives, on each of its expansions. -/
theorem policy_met {p : InferenceOnlyCheck.Protocol.Params} {e : InferenceOnlyCheck.Protocol.Evidence} {out : Json}
    (hd : InferenceOnlyCheck.Desc.evaluate (instanceOf p e) = .ok out) (hc : out.getObjVal? "complies" = .ok (.bool true))
    {P : Prog} (hP : Flock.Program.decode p.program.toUTF8 = .ok P) :
    ∀ (j' : Json) (i : InferenceOnlyCheck.Instance) (d : RegisteredCircuit),
      Expands (instanceOf p e) j' → Parses j' i → LaysOut P d.D → Names i.theta d.θ → Divides i.owner d.𝓘 →
        VerifierOwn p i ∧ PolicyMet d d.𝓘 i.gamma ((i.num : ℝ) / i.den) := by
  intro j' i d hx hi hL hN hD
  have hP' : Flock.Program.decode i.program.toUTF8 = .ok P := expands_program hx hi ▸ hP
  have hR := (InferenceOnlyCheck.Properties.SecurityProofs.DescReports _ _ hd).2 j' hx i P d hi hP' hL hN hD
  exact ⟨verifier_own hx hi, policyMet_of_meets (hR.2.2.1.1 hc)⟩

/-- What the policy's acceptance establishes about its decoded case. -/
theorem policyHeld {p : InferenceOnlyCheck.Protocol.Params} {c : InferenceOnlyCheck.Protocol.Commitment}
    {e : InferenceOnlyCheck.Protocol.Evidence} {pub : Unit} {prof : Json}
    (h : InferenceOnlyCheck.Protocol.check p c e = .accept pub prof) : PolicyHeld p c e := by
  obtain ⟨hn, out, hb, hc⟩ := policy_accept h
  obtain ⟨hd, -⟩ := InferenceOnlyCheck.Properties.SecurityProofs.ThetaSteps.evaluate_ok _ _ _ hb
  obtain ⟨⟨j', hx⟩, hev⟩ := InferenceOnlyCheck.Properties.SecurityProofs.DescAgrees _ _ hd
  obtain ⟨i, P, hpi, hP', D, hL, ⟨θ, hθ, hθin⟩, n, U, hD⟩ :=
    InferenceOnlyCheck.Properties.SecurityProofs.Decodes _ _ (hev j' hx)
  have hP : Flock.Program.decode p.program.toUTF8 = .ok P := expands_program hx hpi ▸ hP'
  let d : RegisteredCircuit := .ofPartition D θ hθin n U
  exact ⟨P, hP, names_program hn hP, InferenceOnlyCheck.Properties.SecurityProofs.ThetaProvenance _ _ _ hb,
    ⟨j', i, d, hx, hpi, hL, hθ, hD, policy_met hd hc hP j' i d hx hpi hL hθ hD⟩, policy_met hd hc hP⟩

/-- What one-stage v1's check at any coins gives on acceptance: its record keeps R2 against the audit's own
verifier. -/
theorem stage_registered_at {p : Spot.OneStage.Params} {c : Spot.OneStage.Commitment}
    {ξ : Option ByteArray} {e : Spot.OneStage.Evidence} {pub : Spot.OneStage.Public}
    {prof : Integrity} (h : Spot.OneStage.checkAt p c ξ e = .accept pub prof) :
    Properties.Spot.R2 c.record p.verifier := by
  unfold Spot.OneStage.checkAt at h
  split at h
  · cases h
  · obtain ⟨samplingDesign, d, n, hreg, -⟩ := Properties.Spot.acceptMeans _ _ _ _ _ h
    exact hreg.2.1

/-- One-stage v2's check at the coins it read accepts only where v1's check accepts at them, on the verdicts its own
run of the verifier computed, in either form: the one statement's run, or the per-member run. -/
theorem stage_zk_at {p : Spot.OneStageZk.Params} {c : Spot.OneStage.Commitment} {ξ : ByteArray}
    {e : Spot.OneStageZk.Evidence} {bs : Blobs} {pub : Spot.OneStage.Public} {prof : Integrity}
    (h : Spot.OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) :
    ∃ r : Spot.OneStageZk.Run,
      Spot.OneStage.checkAt p.base c (some ξ) { e.base with verdicts := r.out } = .accept pub prof := by
  rcases Spot.OneStageZk.checkBlobsAt_accept h with ⟨-, -, h⟩ | ⟨ms, sts, -, -, h⟩
  swap
  · obtain ⟨r, -, -, hc⟩ := Spot.OneStageZk.checkMembersAt_accept h
    exact ⟨r, hc⟩
  unfold Spot.OneStageZk.checkOneAt at h
  dsimp only at h
  split at h
  · cases h
  · rename_i r _
    refine ⟨r, ?_⟩
    generalize Spot.OneStage.checkAt p.base c (some ξ) { e.base with verdicts := r.out } = o at h ⊢
    cases o with
    | accept pub' prof' =>
      cases hr : r.all <;> simp [hr] at h
      obtain ⟨rfl, rfl⟩ := h
      rfl
    | reject s => simp at h
    | undecided s => simp at h

/-- One-stage v2 accepts only where v1's check accepts at the coins it read, on the verdicts its own run of the
verifier computed, so its record keeps R2 against the v1 part of its params' verifier. -/
theorem stage_zk_registered {p : Spot.OneStageZk.Params} {c : Spot.OneStage.Commitment}
    {e : Spot.OneStageZk.Evidence} {bs : Blobs} {pub : Spot.OneStage.Public} {prof : Integrity}
    (h : Spot.OneStageZk.checkBlobs p c e bs = .accept pub prof) :
    Properties.Spot.R2 c.record p.base.verifier := by
  unfold Spot.OneStageZk.checkBlobs Core.atCoins at h
  split at h
  · obtain ⟨_, hr⟩ := stage_zk_at h
    exact stage_registered_at hr
  · cases h

/-- `digestOf` of a commitment that decodes is the digest of what it decodes to. -/
theorem digestOf_ok {P : Protocol} {b : ByteArray} {c : P.Commitment} (h : P.commitment b = .ok c) :
    Properties.Composition.digestOf P b = some (P.digest c) := by
  unfold Properties.Composition.digestOf
  rw [h]
  rfl

/-- A list of length three is three elements. -/
theorem three_of_length {α : Type} {l : List α} (h : l.length = 3) : ∃ a b c, l = [a, b, c] :=
  ⟨_, _, _, List.eq_getElem_of_length_eq_three l h⟩

/-- What an accepted composition of any three components gives: three parameters and three cases, each case carrying
the parameters the composition names for it and decoding to what its component's check accepts. Over components
left abstract, so that no component's code is unfolded. -/
theorem accepted_three {id : String} {P₁ P₂ P₃ : Protocol} {ps : List Named} {c : ByteArray} {ev : List Case}
    {pub prof : Keyed} (h : check id [P₁, P₂, P₃] ps c ev = .accept pub prof) :
    ∃ (n₁ n₂ n₃ : Named) (k₁ k₂ k₃ : Case), ps = [n₁, n₂, n₃] ∧ ev = [k₁, k₂, k₃] ∧
      k₁.id = P₁.id ∧ k₂.id = P₂.id ∧ k₁.params = n₁.params ∧ k₃.params = n₃.params ∧
      Properties.Composition.DecodedAccepted P₁ n₁.params.toUTF8 k₁.commitment k₁.evidence ∧
      Properties.Composition.DecodedAccepted P₂ n₂.params.toUTF8 k₂.commitment k₂.evidence ∧
      Properties.Composition.DecodedAccepted P₃ n₃.params.toUTF8 k₃.commitment k₃.evidence := by
  obtain ⟨⟨h1, h2, -, hpar⟩, hacc, -⟩ := Properties.Composition.acceptMeans _ _ _ _ _ _ _ h
  have hlen : ev.length = 3 := by simpa using congrArg List.length h2
  have hlenp : ps.length = 3 := by simpa using congrArg List.length h1
  obtain ⟨k₁, k₂, k₃, rfl⟩ := three_of_length hlen
  obtain ⟨n₁, n₂, n₃, rfl⟩ := three_of_length hlenp
  simp only [List.map_cons, List.map_nil, List.cons.injEq] at h2
  have m₁ : (P₁, n₁, k₁) ∈ Properties.Composition.triples [P₁, P₂, P₃] [n₁, n₂, n₃] [k₁, k₂, k₃] := by
    simp [Properties.Composition.triples]
  have m₂ : (P₂, n₂, k₂) ∈ Properties.Composition.triples [P₁, P₂, P₃] [n₁, n₂, n₃] [k₁, k₂, k₃] := by
    simp [Properties.Composition.triples]
  have m₃ : (P₃, n₃, k₃) ∈ Properties.Composition.triples [P₁, P₂, P₃] [n₁, n₂, n₃] [k₁, k₂, k₃] := by
    simp [Properties.Composition.triples]
  exact ⟨n₁, n₂, n₃, k₁, k₂, k₃, rfl, rfl, h2.1, h2.2.1, hpar _ m₁, hpar _ m₃, hacc _ m₁, hacc _ m₂, hacc _ m₃⟩

/-- What `strays` finding none gives for the first and third of three components of distinct ids whose commitments
decode: the third's digest is the first's. Over components left abstract, so that no digest is computed. -/
theorem strays_two {P₁ P₂ P₃ : Protocol} {k₁ k₂ k₃ : Case} {c₁ : P₁.Commitment} {c₃ : P₃.Commitment}
    (n₁₃ : (P₁.id == P₃.id) = false) (n₂₃ : (P₂.id == P₃.id) = false) (h₁ : P₁.commitment k₁.commitment = .ok c₁)
    (h₃ : P₃.commitment k₃.commitment = .ok c₃) (h : strays [P₁.id, P₃.id] [P₁, P₂, P₃] [k₁, k₂, k₃] = none) :
    P₃.digest c₃ = P₁.digest c₁ := by
  have e₁ : committedBy P₁.id [P₁, P₂, P₃] [k₁, k₂, k₃] = some (P₁.digest c₁) := by
    simp [committedBy, h₁, Except.toOption]
  have e₃ : committedBy P₃.id [P₁, P₂, P₃] [k₁, k₂, k₃] = some (P₃.digest c₃) := by
    simp [committedBy, n₁₃, n₂₃, h₃, Except.toOption]
  have hm : [P₁.id, P₃.id].mapM (fun s => (committedBy s [P₁, P₂, P₃] [k₁, k₂, k₃]).map (s, ·)) =
      some [(P₁.id, P₁.digest c₁), (P₃.id, P₃.digest c₃)] := by
    simp [e₁, e₃]
  exact strays_none hm h (P₃.id, P₃.digest c₃) (by simp)

/-- `strays_two`'s converse: the first and third of three components of distinct ids whose commitments decode to one
digest leave no stray. -/
theorem strays_two_none {P₁ P₂ P₃ : Protocol} {k₁ k₂ k₃ : Case} {c₁ : P₁.Commitment} {c₃ : P₃.Commitment}
    (n₁₃ : (P₁.id == P₃.id) = false) (n₂₃ : (P₂.id == P₃.id) = false) (h₁ : P₁.commitment k₁.commitment = .ok c₁)
    (h₃ : P₃.commitment k₃.commitment = .ok c₃) (d₃ : P₃.digest c₃ = P₁.digest c₁) :
    strays [P₁.id, P₃.id] [P₁, P₂, P₃] [k₁, k₂, k₃] = none := by
  have e₁ : committedBy P₁.id [P₁, P₂, P₃] [k₁, k₂, k₃] = some (P₁.digest c₁) := by
    simp [committedBy, h₁, Except.toOption]
  have e₃ : committedBy P₃.id [P₁, P₂, P₃] [k₁, k₂, k₃] = some (P₁.digest c₁) := by
    rw [← d₃]; simp [committedBy, n₁₃, n₂₃, h₃, Except.toOption]
  have hm : [P₁.id, P₃.id].mapM (fun s => (committedBy s [P₁, P₂, P₃] [k₁, k₂, k₃]).map (s, ·)) =
      some [(P₁.id, P₁.digest c₁), (P₃.id, P₁.digest c₁)] := by
    simp [e₁, e₃]
  unfold strays
  simp only [hm]
  simp [List.find?, bytes_bne_self]

/-- Remote's check on the caller's blobs is its tie held over `checkSame` over its components on them, the components
named by their ids. -/
theorem remote_check_ids (blobs : Array ByteArray) (ps : List Named) (c : ByteArray) (ev : List Case) :
    (Core.Remote.protocol.on blobs).check ps c ev =
      holdTies [Core.Remote.executionPhases] ev (checkSame "verity/remote/v0"
        [InferenceOnlyCheck.Protocol.protocol.id, (VStar.Recursive.Sampled.protocol.onBlobs (Blobs.ofArray blobs)).id]
        [InferenceOnlyCheck.Protocol.protocol, PouwVerifier.Period.protocol,
          VStar.Recursive.Sampled.protocol.onBlobs (Blobs.ofArray blobs)] ps c ev) := by
  rw [remote_check, id_policy, id_sampled]

theorem ne_policy_stage (bs : Blobs) :
    (InferenceOnlyCheck.Protocol.protocol.id == (VStar.Recursive.Sampled.protocol.onBlobs bs).id) = false := by
  rw [id_policy, id_sampled]; decide

theorem ne_monitoringPeriod_stage (bs : Blobs) :
    (PouwVerifier.Period.protocol.id == (VStar.Recursive.Sampled.protocol.onBlobs bs).id) = false := by
  rw [id_monitoringPeriod, id_sampled]; decide

/-- What remote's tie gives once the policy's commitment decodes: every Commit record the period's windows name, when
they decode, has the policy's record's digest. -/
theorem executionPhases_tied {kI kW kS : Case} {cI : InferenceOnlyCheck.Protocol.Commitment}
    (hI : kI.id = InferenceOnlyCheck.Protocol.protocol.id) (hW : kW.id = PouwVerifier.Period.protocol.id)
    (hcI : InferenceOnlyCheck.Protocol.protocol.commitment kI.commitment = .ok cI)
    (ht : Properties.Composition.Tied Core.Remote.executionPhases [kI, kW, kS]) :
    ∀ ds, Core.Remote.executionPhaseRecords kW = some ds → ∀ d ∈ ds, d = InferenceOnlyCheck.Protocol.protocol.digest cI := by
  intro ds hds d hd
  rw [id_policy] at hI
  rw [id_monitoringPeriod] at hW
  have nI : Properties.Composition.named [kI, kW, kS] (InferenceOnlyCheck.Protocol.protocolId, Core.Remote.policyRecord) =
      some [InferenceOnlyCheck.Protocol.protocol.digest cI] := by
    have hid : (kI.id == InferenceOnlyCheck.Protocol.protocolId) = true := by rw [hI]; decide
    simp [Properties.Composition.named, List.find?, hid, Core.Remote.policyRecord, hcI, Except.toOption]
    exact ⟨cI, rfl, rfl⟩
  have nW : Properties.Composition.named [kI, kW, kS] (PouwVerifier.Period.protocolId, Core.Remote.executionPhaseRecords) =
      some ds := by
    have hI' : (kI.id == PouwVerifier.Period.protocolId) = false := by rw [hI]; decide
    have hW' : (kW.id == PouwVerifier.Period.protocolId) = true := by rw [hW]; decide
    simp [Properties.Composition.named, List.find?, hI', hW', hds]
  have mI : (InferenceOnlyCheck.Protocol.protocolId, Core.Remote.policyRecord) ∈ Core.Remote.executionPhases.names := by
    simp [Core.Remote.executionPhases]
  have mW : (PouwVerifier.Period.protocolId, Core.Remote.executionPhaseRecords) ∈ Core.Remote.executionPhases.names := by
    simp [Core.Remote.executionPhases]
  rcases ht with ⟨n, hn, e⟩ | ⟨r, hr⟩
  · simp only [Core.Remote.executionPhases, List.mem_cons, List.not_mem_nil, or_false] at hn
    rcases hn with rfl | rfl
    · rw [nI] at e; cases e
    · rw [nW] at e; cases e
  · obtain ⟨xs, hx, hxr⟩ := hr _ mI
    obtain ⟨ys, hy, hyr⟩ := hr _ mW
    rw [nI] at hx
    rw [nW] at hy
    cases hx
    cases hy
    rw [hyr d hd, ← hxr _ (List.mem_singleton_self _)]

/-! ## The caller's check's steps -/

/-- The declared day's code (`VStar.Recursive.Caller.declaredCode`) is none exactly when `DeclaredDay` holds. -/
theorem declaredCode_none_iff (rec : Json) (v : Spot.Docket.Verifier) :
    (VStar.Recursive.Caller.declaredCode (Spot.getD rec "program") v.declared).isSome = false ↔
      Properties.Spot.DeclaredDay rec v := by
  unfold Properties.Spot.DeclaredDay VStar.Recursive.Caller.declaredCode
  rcases v.declared with _ | ⟨day, ds⟩
  · simp
  · rcases h : Spot.Docket.declaredOf day ds with _ | ⟨p, _ | ⟨q, l⟩⟩
    · simp [h]
    · by_cases hp : Spot.jeq (Spot.getD rec "program") (.str p) = true <;> simp [h, hp]
    · simp [h]

/-- The pool chain (`VStar.Recursive.Caller.poolChained`) holds exactly when `PoolChained` does. -/
theorem poolChained_iff (rec : Json) (v : Spot.Docket.Verifier) :
    VStar.Recursive.Caller.poolChained rec v = true ↔ Properties.Spot.PoolChained rec v := by
  unfold VStar.Recursive.Caller.poolChained Properties.Spot.PoolChained
  rcases hp : v.pool with _ | p
  · simp
  · rcases he : Spot.Docket.entryOf rec p.pin with _ | e <;>
      rcases ho : Spot.Docket.poolOpen rec v p with _ | ⟨d, r⟩ <;> simp [he, ho]
    all_goals exact ⟨fun h => ⟨d, r, ⟨rfl, rfl⟩, h⟩, fun ⟨_, _, ⟨h1, h2⟩, h⟩ => by subst h1 h2; exact h⟩

/-- The pool's codes (`VStar.Recursive.Caller.poolMarks`) add none exactly when the window names the pool's reset, week
and spent mark and no reset falls where the week's is spent before it. -/
theorem poolMarks_nil (w : Json) (pool : Option Spot.Docket.Pool) (log : Array Json) (c : Array String) :
    VStar.Recursive.Caller.poolMarks w pool log c = #[] ↔ c = #[] ∧
      Spot.jeq (Spot.getD w "pool_reset") (match pool with | some p => .bool p.reset | none => .null) = true ∧
      Spot.jeq (Spot.getD w "pool_week") (match pool with | some p => Lean.toJson p.week | none => .null) = true ∧
      Spot.jeq (Spot.getD w "pool_spent")
        (match pool with | some p => .bool (p.reset || p.spentBefore log) | none => .null) = true ∧
      (match pool with | some p => !(p.reset && p.spentBefore log) | none => true) = true := by
  rw [VStar.Recursive.Caller.poolMarks, Properties.Spot.kept_nil, Properties.Spot.kept_nil, Properties.Spot.kept_nil,
    Properties.Spot.kept_nil]
  exact ⟨fun ⟨⟨⟨⟨a, r⟩, wk⟩, s⟩, o⟩ => ⟨a, r, wk, s, o⟩, fun ⟨a, r, wk, s, o⟩ => ⟨⟨⟨⟨a, r⟩, wk⟩, s⟩, o⟩⟩

/-- The registered compiler's codes (`VStar.Recursive.Caller.compilerMarks`) add none exactly when the window names the
log's compiler and advice within its length. -/
theorem compilerMarks_nil (w : Json) (b : Option (Option (String × Nat))) (c : Array String) :
    VStar.Recursive.Caller.compilerMarks w b c = #[] ↔ c = #[] ∧ Spot.Docket.compilerOk w b = true ∧
      Spot.Docket.adviceOk w b = true := by
  rw [VStar.Recursive.Caller.compilerMarks, Properties.Spot.kept_nil, Properties.Spot.kept_nil, and_assoc]

/-- The declared day's codes (`VStar.Recursive.Caller.declaredMarks`) add none exactly when `DeclaredDay` holds. -/
theorem declaredMarks_nil (rec : Json) (v : Spot.Docket.Verifier) (c : Array String) :
    VStar.Recursive.Caller.declaredMarks (Spot.getD rec "program") v.declared c = #[] ↔
      c = #[] ∧ Properties.Spot.DeclaredDay rec v := by
  rw [← declaredCode_none_iff]
  unfold VStar.Recursive.Caller.declaredMarks
  cases VStar.Recursive.Caller.declaredCode (Spot.getD rec "program") v.declared with
  | none => simp
  | some code => simp [Properties.Spot.addCode_ne_nil]

/-- The binding's codes (`VStar.Recursive.Caller.bindingMarks`) add none exactly when the window names the stream and
the prover of the binding, if any. -/
theorem bindingMarks_nil (w : Json) (b : Option (String × String)) (c : Array String) :
    VStar.Recursive.Caller.bindingMarks w b c = #[] ↔ c = #[] ∧
      (∀ sp, b = some sp → Spot.jeq (Spot.getD w "stream") (.str sp.1) = true ∧
        Spot.jeq (Spot.getD w "prover") (.str sp.2) = true) := by
  rcases b with _ | ⟨stream, prover⟩
  · simp [VStar.Recursive.Caller.bindingMarks]
  · rw [VStar.Recursive.Caller.bindingMarks, Properties.Spot.kept_nil, Properties.Spot.kept_nil]
    constructor
    · rintro ⟨⟨a, hs⟩, hp⟩
      exact ⟨a, fun sp hsp => by cases hsp; exact ⟨hs, hp⟩⟩
    · rintro ⟨a, h⟩
      exact ⟨⟨a, (h _ rfl).1⟩, (h _ rfl).2⟩

/-- The two codes the caller's check adds for a verifier that expects values registered (`R7-pool-chain`,
`R7-output-refused`) are none exactly when, for such a verifier, `PoolChained` and `OutputPortsAccepted` hold. -/
theorem v2Marks_iff (rec : Json) (v : Spot.Docket.Verifier) :
    ((!v.v2 || VStar.Recursive.Caller.poolChained rec v) = true ∧ (!v.v2 || Spot.Docket.portsAccepted v) = true) ↔
      (Properties.Spot.ExpectsV2 v → Properties.Spot.PoolChained rec v ∧ Properties.Spot.OutputPortsAccepted v) := by
  rw [← poolChained_iff, ← Properties.Spot.portsAccepted_iff]
  unfold Properties.Spot.ExpectsV2 Spot.Docket.Verifier.v2
  cases v.registered.isEmpty && v.certified.isEmpty <;> simp

/-- **`CallerCheckMeans`**: the caller's check finds nothing exactly where the record keeps the caller's facts. -/
theorem callerCheckMeans : CallerCheckMeans := by
  intro rec v
  simp only [VStar.Recursive.Caller.check]
  rw [Properties.Spot.kept_nil, Properties.Spot.kept_nil, and_assoc, v2Marks_iff, compilerMarks_nil, poolMarks_nil,
    Properties.Spot.kept_nil, bindingMarks_nil, declaredMarks_nil]
  unfold Properties.Spot.CallerHeld Properties.Spot.RunCompiler
  exact ⟨fun ⟨⟨⟨⟨⟨⟨_, dd⟩, bd⟩, ss⟩, r, wk, sp, o⟩, ck, ad⟩, v2⟩ => ⟨dd, bd, ss, r, wk, sp, o, ⟨ck, ad⟩, v2⟩,
    fun ⟨dd, bd, ss, r, wk, sp, o, ⟨ck, ad⟩, v2⟩ => ⟨⟨⟨⟨⟨⟨rfl, dd⟩, bd⟩, ss⟩, r, wk, sp, o⟩, ck, ad⟩, v2⟩⟩

/-- **`CallerCheckSound`**, `CallerCheckMeans`' forward half. -/
theorem callerCheckSound_of_means (hm : CallerCheckMeans) : CallerCheckSound := fun rec v => (hm rec v).1

/-- `CallerCheckSound`, from `callerCheckMeans`. -/
theorem callerCheckSound : CallerCheckSound := callerCheckSound_of_means callerCheckMeans

/-- What SPOT's accepted case gives, on any blobs: `StageHeld`, in the one-stage form, where the caller's
check finds nothing only where the record keeps the caller's facts (`CallerCheckSound`). -/
theorem stage_held (hm : CallerCheckSound) {k : Case} {n : Named} {bs : Blobs} (hp : k.params = n.params)
    (h : Properties.Composition.DecodedAccepted (VStar.Recursive.Sampled.protocol.onBlobs bs) n.params.toUTF8
      k.commitment k.evidence) : StageHeld k := by
  obtain ⟨p, c, e, hp', hc, -, pub, prof, hchk⟩ := h
  have hp'' : (canonicalJson k.params.toUTF8 >>= VStar.Recursive.Sampled.paramsOfJson) = .ok p := by
    rw [hp]; exact hp'
  have hc' : Spot.OneStage.commitmentOf k.commitment = .ok c := hc
  have hchk' : VStar.Recursive.Sampled.checkBlobs p c e bs = .accept pub prof := hchk
  clear hp' hchk
  rcases p with p | q <;> rcases e with e | e
  · obtain ⟨hcall, hz⟩ := VStar.Recursive.Caller.gate_accept
      (show VStar.Recursive.Caller.gate c.record p.base.verifier (Spot.OneStageZk.checkBlobs p c e bs) =
        .accept pub prof from hchk')
    exact ⟨p, c, hp'', hc', stage_zk_registered hz, hm _ _ hcall⟩
  · simp [VStar.Recursive.Sampled.checkBlobs] at hchk'
  · simp [VStar.Recursive.Sampled.checkBlobs] at hchk'
  · exact absurd hchk' (VStar.Recursive.SampledTwo.checkBlobs_refuses VStar.Recursive.SampledTwo.form_off q c e bs
      pub prof)

/-- What remote's acceptance gives: the three cases in order, one digest for the policy's and SPOT's
records, which every Commit record the period's windows name has, the policy's case decoding to what the policy's
check accepts, and SPOT's case decoding to what its check accepts on the blobs. -/
theorem remote_accept {blobs : Array ByteArray} {ps : List Named} {c : ByteArray} {ev : List Case} {pub prof : Keyed}
    (h : (Core.Remote.protocol.on blobs).check ps c ev = .accept pub prof) :
    ∃ (kI kW kS : Case) (pI : InferenceOnlyCheck.Protocol.Params) (cI : InferenceOnlyCheck.Protocol.Commitment)
      (eI : InferenceOnlyCheck.Protocol.Evidence), ev = [kI, kW, kS] ∧
      Properties.Composition.digestOf InferenceOnlyCheck.Protocol.protocol kI.commitment = some (InferenceOnlyCheck.Protocol.protocol.digest cI) ∧
      (∀ ds, Core.Remote.executionPhaseRecords kW = some ds → ∀ d ∈ ds, d = InferenceOnlyCheck.Protocol.protocol.digest cI) ∧
      Properties.Composition.digestOf Spot.OneStage.protocol kS.commitment = some (InferenceOnlyCheck.Protocol.protocol.digest cI) ∧
      (canonicalJson kI.params.toUTF8 >>= InferenceOnlyCheck.Protocol.paramsOf) = .ok pI ∧
      InferenceOnlyCheck.Protocol.commitmentOf kI.commitment = .ok cI ∧ InferenceOnlyCheck.Protocol.evidenceOf kI.evidence = .ok eI ∧
      (∃ pub' prof', InferenceOnlyCheck.Protocol.check pI cI eI = .accept pub' prof') ∧
      ∃ nS : Named, kS.params = nS.params ∧ Properties.Composition.DecodedAccepted
        (VStar.Recursive.Sampled.protocol.onBlobs (Blobs.ofArray blobs)) nS.params.toUTF8 kS.commitment kS.evidence := by
  rw [remote_check_ids] at h
  obtain ⟨hsame, hties⟩ := Properties.Composition.acceptTied _ _ _ _ _ h
  obtain ⟨hcheck, hstrays⟩ := checkSame_accept hsame
  obtain ⟨nI, nW, nS, kI, kW, kS, -, rfl, hidI, hidW, hpI, hpS, ⟨pI, cI, eI, hpI', hcI, heI, hchkI⟩, -, hS⟩ :=
    accepted_three hcheck
  have hS₀ := hS
  obtain ⟨pS, cS, eS, -, hcS, -, -⟩ := hS₀
  have hdS := strays_two (P₁ := InferenceOnlyCheck.Protocol.protocol) (P₂ := PouwVerifier.Period.protocol)
    (P₃ := VStar.Recursive.Sampled.protocol.onBlobs (Blobs.ofArray blobs)) (ne_policy_stage _) (ne_monitoringPeriod_stage _)
    hcI hcS hstrays
  have hp : (canonicalJson kI.params.toUTF8 >>= InferenceOnlyCheck.Protocol.paramsOf) = .ok pI := by rw [hpI]; exact hpI'
  refine ⟨kI, kW, kS, pI, cI, eI, rfl, digestOf_ok hcI,
    executionPhases_tied hidI hidW hcI (hties _ (List.mem_singleton_self _)),
    (digestOf_sampled _ _).symm.trans ((digestOf_ok hcS).trans (congrArg some hdS)), hp, hcI, heI, hchkI,
    nS, hpS, hS⟩

/-- `AcceptInferenceOnly`, on the proved caller's check (`callerCheckSound`). -/
theorem acceptInferenceOnly : AcceptInferenceOnly := by
  intro blobs ps c ev pub prof h
  obtain ⟨kI, kW, kS, pI, cI, eI, hev, hdI, hdW, hdS, hp, hc, he, ⟨pub', prof', hchk⟩, nS, hpS, hS⟩ := remote_accept h
  exact ⟨kI, kW, kS, InferenceOnlyCheck.Protocol.protocol.digest cI, hev, hdI, hdW, hdS, ⟨pI, cI, eI, hp, hc, he, policyHeld hchk⟩,
    stage_held callerCheckSound hpS hS⟩

/-- **`AcceptInferenceOnly`'s nontrivial lemma**: its conclusion fails at no cases at all, `ev = []`, which a check that
accepts every input accepts: they are not the three subprotocols' cases. -/
theorem acceptInferenceOnly_nontrivial : ¬ InferenceOnlyHeld [] := by
  rintro ⟨kI, kW, kS, r, hev, -⟩
  cases hev

theorem acceptComplies : AcceptComplies := by
  intro blobs ps c ev pub prof h
  obtain ⟨kI, kW, kS, pI, cI, eI, hev, -, -, -, hp, hc, he, ⟨pub', prof', hchk⟩, -⟩ := remote_accept h
  obtain ⟨hn, out, hb, hcomp⟩ := policy_accept hchk
  obtain ⟨hd, -⟩ := InferenceOnlyCheck.Properties.SecurityProofs.ThetaSteps.evaluate_ok _ _ _ hb
  obtain ⟨P, hP, hname, -, ⟨j', i, d, hx, hpi, hL, hθ, -, hown, hmet⟩, -⟩ := policyHeld hchk
  refine ⟨kI, kW, kS, pI, cI, eI, P, hev, hp, hc, he, hP, hname, ⟨j', i, d, hx, hpi, hL, hθ, hown, hmet.2.1⟩, ?_⟩
  intro j' i d hx hi hL hN
  have hP' : Flock.Program.decode i.program.toUTF8 = .ok P := expands_program hx hi ▸ hP
  exact ⟨verifier_own hx hi, ((InferenceOnlyCheck.Properties.SecurityProofs.DescWitnesses _ _ hd).2 j' hx i P d hi hP' hL hN hcomp).1⟩

/-- **`AcceptComplies`'s nontrivial lemma**, as `AcceptInferenceOnly`'s: its conclusion fails at `ev = []`, which a check
that accepts every input accepts. -/
theorem acceptComplies_nontrivial : ¬ CompliesHeld [] := by
  rintro ⟨kI, kW, kS, p, cr, e, P, hev, -⟩
  cases hev

end Properties.Remote.SecurityProofs
