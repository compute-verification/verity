import Properties.Remote
import Proofs.Properties.Composition
import Proofs.Properties.Nci.DescVerdict
import Proofs.Properties.Nci.Baseline
import Proofs.Properties.Nci.BaselineSteps
import Proofs.Properties.Nci
import Proofs.Properties.Nci.Agrees
import Proofs.Nci.FitWitness
import Proofs.Properties.SampledProofs
import Proofs.Nci.CompliesCK
import Proofs.Core.Expand
import Std.Data.TreeMap.Raw.Lemmas

/-!
# Remote's acceptance and the inference-only policy, proved

Remote's check is `Core.Composition.holdTies` of its tie over `Core.Composition.checkSame` over its components on the
caller's blobs (`remote_check`), so an acceptance is `checkSame`'s with the tie held (`Properties.Composition.acceptTied`),
and `checkSame`'s is `Core.Composition.check`'s (`checkSame_accept`), which `Properties.Composition.acceptMeans` reduces
to each component accepting its own case, with the one-stage audit's record the policy's (`strays_two`) and every
Commit record the period's windows name the policy's record too (`windows_tied`). The policy's
acceptance is `NciCheck.Baseline.evaluate`'s with the record naming the circuit (`policy_accept`), which is
`NciCheck.Desc.evaluate`'s and θ's provenance (`BaselineSteps.evaluate_ok`), whose verdict `DescReports`,
`DescWitnesses` and `BaselineProvenance` read; `Nci.SecurityProofs.CompliesCK`, `ExpandComplies` and
`Verity.Circuit.expand_boolean` give the padded `(c, k)`-partitions. The parse, declaration and division an acceptance
exhibits come from `DescAgrees` (the verdict is the gate-by-gate check's on an expansion) and `Decodes` (that check's
verdicts are about a parse, a layout, a baseline and a division).

The policy's existence clause is neither empty nor trivial: `policyMet_honest` exhibits a declaration and a division into
nonempty inference blocks that meet the policy, and `policyMet_nontrivial` one the policy refuses at every `α`. Both
conclusions fail at the cases a check that accepts every input accepts, `ev = []` (`acceptInferenceOnly_nontrivial`,
`acceptComplies_nontrivial`).

The digests' tie and the cases' decoding are proved over three components left abstract (`accepted_three`,
`strays_two`), then read at remote's, so that no subprotocol's decoder or digest is unfolded by `simp`.
-/

namespace Properties.Remote.SecurityProofs

open Core Core.Composition Verity
open Lean (Json)
open Flock.Program (Prog)
open NciCheck.Properties (Parses Expands LaysOut Names Divides Meets ProvenanceHeld Reported Witnessed)
open NciCheck.Protocol (instanceOf trustedOf)

private theorem getObjVal?_mkObj {l : List (String × Json)} (hd : l.Pairwise (fun a b => ¬ compare a.1 b.1 = .eq))
    {k : String} {v : Json} (hm : (k, v) ∈ l) : (Json.mkObj l).getObjVal? k = .ok v := by
  simp only [Json.mkObj, Json.getObjVal?, Std.TreeMap.Raw.get?_eq_getElem?]
  rw [Std.TreeMap.Raw.getElem?_ofList_of_mem (Std.ReflCmp.compare_self) hd hm]
  rfl

theorem instance_program (p : NciCheck.Protocol.Params) (e : NciCheck.Protocol.Evidence) :
    (instanceOf p e).getObjVal? "program" = .ok (.str p.program) :=
  getObjVal?_mkObj (by simp) (by simp)

theorem instance_gamma (p : NciCheck.Protocol.Params) (e : NciCheck.Protocol.Evidence) :
    (instanceOf p e).getObjVal? "gamma" = .ok p.gamma :=
  getObjVal?_mkObj (by simp) (by simp)

theorem instance_alpha (p : NciCheck.Protocol.Params) (e : NciCheck.Protocol.Evidence) :
    (instanceOf p e).getObjVal? "alpha" = .ok p.alpha :=
  getObjVal?_mkObj (by simp) (by simp)

/-- Two parses of one instance agree on its program, `Γ` and `α`. -/
theorem parses_same {j : Json} {i i' : NciCheck.Instance} (h : Parses j i) (h' : Parses j i') :
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
theorem verifier_own {p : NciCheck.Protocol.Params} {e : NciCheck.Protocol.Evidence} {j' : Json}
    {i : NciCheck.Instance} (hx : Expands (instanceOf p e) j') (hi : Parses j' i) : VerifierOwn p i := by
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

/-- An expansion of the policy's instance, and what it parses to, carry the declared circuit's text. -/
theorem expands_program {p : NciCheck.Protocol.Params} {e : NciCheck.Protocol.Evidence} {j' : Json}
    {i : NciCheck.Instance} (hx : Expands (instanceOf p e) j') (hi : Parses j' i) : i.program = p.program := by
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
      holdTies [Core.Remote.windows] ev (checkSame "verity/remote/v0"
        ["verity/inference-only/v0", "verity/sampled-proofs/one-stage/v1"]
        [NciCheck.Protocol.protocol, PouwCheck.Period.protocol, SampledProofs.OneStage.protocol] ps c ev) := rfl

theorem id_policy : NciCheck.Protocol.protocol.id = "verity/inference-only/v0" := rfl
theorem id_period : PouwCheck.Period.protocol.id = "verity/pouw/period/v1" := rfl
theorem id_oneStage : SampledProofs.OneStage.protocol.id = "verity/sampled-proofs/one-stage/v1" := rfl

/-- What the policy's check accepts: a record naming the declared circuit, and a verdict of compliance of the check of
record. -/
theorem policy_accept {p : NciCheck.Protocol.Params} {c : NciCheck.Protocol.Commitment}
    {e : NciCheck.Protocol.Evidence} {pub : Unit} {prof : Json}
    (h : NciCheck.Protocol.check p c e = .accept pub prof) :
    NciCheck.Protocol.names p c = true ∧ ∃ out, NciCheck.Baseline.evaluate (instanceOf p e) (trustedOf p c) = .ok out ∧
      out.getObjVal? "complies" = .ok (.bool true) := by
  unfold NciCheck.Protocol.check at h
  by_cases hn : NciCheck.Protocol.names p c = true
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
theorem names_program {p : NciCheck.Protocol.Params} {c : NciCheck.Protocol.Commitment} {P : Prog}
    (hn : NciCheck.Protocol.names p c = true) (hP : Flock.Program.decode p.program.toUTF8 = .ok P) :
    NciCheck.Baseline.strAt c.record "program" = some (Flock.hexOf P.sha512) := by
  unfold NciCheck.Protocol.names at hn
  rw [hP] at hn
  simpa using hn

/-- `PolicyMet` from its first clause, every inference block meeting both bounds, at `α ≥ 0`: the rest follows
(`Nci.Complies`, and the padded partitions by `CompliesCK`, `ExpandComplies` and `expand_boolean`). -/
theorem policyMet_of_meets {d : Declaration} {n : ℕ} {U : Partition d.D n} {Γ : ℕ} {α : ℝ} (hα : 0 ≤ α)
    (h : ∀ u, Meets d U Γ α u) : PolicyMet d U Γ α := by
  have hcomp : Nci.Complies Γ α d := ⟨n, U, h⟩
  refine ⟨h, hcomp, fun hB => Nci.SecurityProofs.CompliesCK _ _ d hB hcomp, ?_⟩
  intro W hfan hw hW
  exact Nci.SecurityProofs.CompliesCK _ _ d.expand (d.D.expand_boolean hfan)
    (Nci.SecurityProofs.ExpandComplies Γ W _ d hα hw hW hcomp)

/-- **Honest witness of the policy's existence clause**: a declaration and a division into nonempty inference blocks
that meet the policy, `Nci.SecurityProofs.FitWitness`'s five-gate circuit in its two one-gate windows, at `Γ = 1` and
`α = 3`. -/
theorem policyMet_honest : ∃ (d : Declaration) (n : ℕ) (U : Partition d.D n) (Γ : ℕ) (α : ℝ),
    0 < n ∧ (∀ u, (U.gates u).Nonempty) ∧ PolicyMet d U Γ α := by
  have hs : ∀ u, Nci.SecurityProofs.FitWitness.decl.D.size (Nci.SecurityProofs.FitWitness.windows.gates u) = 1 := by
    decide +kernel
  refine ⟨Nci.SecurityProofs.FitWitness.decl, 2, Nci.SecurityProofs.FitWitness.windows, 1, 3, by norm_num,
    by decide +kernel, policyMet_of_meets (by norm_num) fun u => ⟨(hs u).le, ?_⟩⟩
  rw [hs u, Nat.cast_one, mul_one]
  exact Nci.SecurityProofs.FitWitness.reads_le u

/-- **Nontrivial witness of the policy's existence clause**: the policy refuses a declaration and a division into
nonempty inference blocks, the same windows at `Γ = 0`, at every `α`: each block is a gate of width 1. -/
theorem policyMet_nontrivial : ∃ (d : Declaration) (n : ℕ) (U : Partition d.D n),
    (∀ u, (U.gates u).Nonempty) ∧ ∀ α : ℝ, ¬ PolicyMet d U 0 α := by
  refine ⟨Nci.SecurityProofs.FitWitness.decl, 2, Nci.SecurityProofs.FitWitness.windows, by decide +kernel,
    fun α h => ?_⟩
  have hs : Nci.SecurityProofs.FitWitness.decl.D.size (Nci.SecurityProofs.FitWitness.windows.gates 0) = 1 := by
    decide +kernel
  have h0 := (h.1 0).1
  omega

/-- What a verdict of compliance on the policy's instance gives, on each of its expansions. -/
theorem policy_met {p : NciCheck.Protocol.Params} {e : NciCheck.Protocol.Evidence} {out : Json}
    (hd : NciCheck.Desc.evaluate (instanceOf p e) = .ok out) (hc : out.getObjVal? "complies" = .ok (.bool true))
    {P : Prog} (hP : Flock.Program.decode p.program.toUTF8 = .ok P) :
    ∀ (j' : Json) (i : NciCheck.Instance) (d : Declaration) (n : ℕ) (U : Partition d.D n),
      Expands (instanceOf p e) j' → Parses j' i → LaysOut P d.D → Names i.theta d.θ → Divides i.owner U →
        VerifierOwn p i ∧ PolicyMet d U i.gamma ((i.num : ℝ) / i.den) := by
  intro j' i d n U hx hi hL hN hD
  have hP' : Flock.Program.decode i.program.toUTF8 = .ok P := expands_program hx hi ▸ hP
  have hR := (NciCheck.Properties.SecurityProofs.DescReports _ _ hd).2 j' hx i P d n U hi hP' hL hN hD
  exact ⟨verifier_own hx hi,
    policyMet_of_meets (div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)) (hR.2.2.1.1 hc)⟩

/-- What the policy's acceptance establishes about its decoded case. -/
theorem policyHeld {p : NciCheck.Protocol.Params} {c : NciCheck.Protocol.Commitment}
    {e : NciCheck.Protocol.Evidence} {pub : Unit} {prof : Json}
    (h : NciCheck.Protocol.check p c e = .accept pub prof) : PolicyHeld p c e := by
  obtain ⟨hn, out, hb, hc⟩ := policy_accept h
  obtain ⟨hd, -⟩ := NciCheck.Properties.SecurityProofs.BaselineSteps.evaluate_ok _ _ _ hb
  obtain ⟨⟨j', hx⟩, hev⟩ := NciCheck.Properties.SecurityProofs.DescAgrees _ _ hd
  obtain ⟨i, P, hpi, hP', D, hL, ⟨θ, hθ, hθin⟩, n, U, hD⟩ :=
    NciCheck.Properties.SecurityProofs.Decodes _ _ (hev j' hx)
  have hP : Flock.Program.decode p.program.toUTF8 = .ok P := expands_program hx hpi ▸ hP'
  let d : Declaration := ⟨D, θ, hθin⟩
  exact ⟨P, hP, names_program hn hP, NciCheck.Properties.SecurityProofs.BaselineProvenance _ _ _ hb,
    ⟨j', i, d, n, U, hx, hpi, hL, hθ, hD, policy_met hd hc hP j' i d n U hx hpi hL hθ hD⟩, policy_met hd hc hP⟩

/-- What the one-stage audit's acceptance gives: its record keeps R2 against the audit's own verifier. -/
theorem stage_registered {p : SampledProofs.OneStage.Params} {c : SampledProofs.OneStage.Commitment}
    {e : SampledProofs.OneStage.Evidence} {pub : SampledProofs.OneStage.Public} {prof : Integrity}
    (h : SampledProofs.OneStage.checkLive p c e = .accept pub prof) :
    Properties.SampledProofs.R2 c.record p.verifier := by
  unfold SampledProofs.OneStage.checkLive Core.atCoins at h
  split at h
  · unfold SampledProofs.OneStage.checkAt at h
    beta_reduce at h
    split at h
    · cases h
    · obtain ⟨law, d, n, hreg, -⟩ := Properties.SampledProofs.acceptMeans _ _ _ _ _ h
      exact hreg.2.1
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
      holdTies [Core.Remote.windows] ev (checkSame "verity/remote/v0"
        [NciCheck.Protocol.protocol.id, SampledProofs.OneStage.protocol.id]
        [NciCheck.Protocol.protocol, PouwCheck.Period.protocol, SampledProofs.OneStage.protocol] ps c ev) := by
  rw [remote_check, id_policy, id_oneStage]

theorem ne_policy_stage : (NciCheck.Protocol.protocol.id == SampledProofs.OneStage.protocol.id) = false := by
  rw [id_policy, id_oneStage]; decide

theorem ne_period_stage : (PouwCheck.Period.protocol.id == SampledProofs.OneStage.protocol.id) = false := by
  rw [id_period, id_oneStage]; decide

/-- What remote's tie gives once the policy's commitment decodes: every Commit record the period's windows name, when
they decode, has the policy's record's digest. -/
theorem windows_tied {kI kW kS : Case} {cI : NciCheck.Protocol.Commitment}
    (hI : kI.id = NciCheck.Protocol.protocol.id) (hW : kW.id = PouwCheck.Period.protocol.id)
    (hcI : NciCheck.Protocol.protocol.commitment kI.commitment = .ok cI)
    (ht : Properties.Composition.Tied Core.Remote.windows [kI, kW, kS]) :
    ∀ ds, Core.Remote.windowRecords kW = some ds → ∀ d ∈ ds, d = NciCheck.Protocol.protocol.digest cI := by
  intro ds hds d hd
  rw [id_policy] at hI
  rw [id_period] at hW
  have nI : Properties.Composition.named [kI, kW, kS] (NciCheck.Protocol.protocolId, Core.Remote.policyRecord) =
      some [NciCheck.Protocol.protocol.digest cI] := by
    have hid : (kI.id == NciCheck.Protocol.protocolId) = true := by rw [hI]; decide
    simp [Properties.Composition.named, List.find?, hid, Core.Remote.policyRecord, hcI, Except.toOption]
    exact ⟨cI, rfl, rfl⟩
  have nW : Properties.Composition.named [kI, kW, kS] (PouwCheck.Period.protocolId, Core.Remote.windowRecords) =
      some ds := by
    have hI' : (kI.id == PouwCheck.Period.protocolId) = false := by rw [hI]; decide
    have hW' : (kW.id == PouwCheck.Period.protocolId) = true := by rw [hW]; decide
    simp [Properties.Composition.named, List.find?, hI', hW', hds]
  have mI : (NciCheck.Protocol.protocolId, Core.Remote.policyRecord) ∈ Core.Remote.windows.names := by
    simp [Core.Remote.windows]
  have mW : (PouwCheck.Period.protocolId, Core.Remote.windowRecords) ∈ Core.Remote.windows.names := by
    simp [Core.Remote.windows]
  rcases ht with ⟨n, hn, e⟩ | ⟨r, hr⟩
  · simp only [Core.Remote.windows, List.mem_cons, List.not_mem_nil, or_false] at hn
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

/-- What the one-stage audit's accepted case gives: `StageHeld`. -/
theorem stage_held {k : Case} {n : Named} (hp : k.params = n.params)
    (h : Properties.Composition.DecodedAccepted SampledProofs.OneStage.protocol n.params.toUTF8
      k.commitment k.evidence) : StageHeld k := by
  obtain ⟨p, c, e, hp', hc, -, pub, prof, hchk⟩ := h
  have hp'' : (canonicalJson n.params.toUTF8 >>= SampledProofs.OneStage.paramsOfJson) = .ok p := hp'
  have hc' : SampledProofs.OneStage.commitmentOf k.commitment = .ok c := hc
  have hchk' : SampledProofs.OneStage.checkLive p c e = .accept pub prof := hchk
  exact ⟨p, c, by rw [hp]; exact hp'', hc', stage_registered hchk'⟩

/-- What remote's acceptance gives: the three cases in order, one digest for the policy's and the one-stage audit's
records, which every Commit record the period's windows name has, and the policy's case decoding to what the policy's
check accepts. -/
theorem remote_accept {blobs : Array ByteArray} {ps : List Named} {c : ByteArray} {ev : List Case} {pub prof : Keyed}
    (h : (Core.Remote.protocol.on blobs).check ps c ev = .accept pub prof) :
    ∃ (kI kW kS : Case) (pI : NciCheck.Protocol.Params) (cI : NciCheck.Protocol.Commitment)
      (eI : NciCheck.Protocol.Evidence), ev = [kI, kW, kS] ∧
      Properties.Composition.digestOf NciCheck.Protocol.protocol kI.commitment = some (NciCheck.Protocol.protocol.digest cI) ∧
      (∀ ds, Core.Remote.windowRecords kW = some ds → ∀ d ∈ ds, d = NciCheck.Protocol.protocol.digest cI) ∧
      Properties.Composition.digestOf SampledProofs.OneStage.protocol kS.commitment = some (NciCheck.Protocol.protocol.digest cI) ∧
      (canonicalJson kI.params.toUTF8 >>= NciCheck.Protocol.paramsOf) = .ok pI ∧
      NciCheck.Protocol.commitmentOf kI.commitment = .ok cI ∧ NciCheck.Protocol.evidenceOf kI.evidence = .ok eI ∧
      (∃ pub' prof', NciCheck.Protocol.check pI cI eI = .accept pub' prof') ∧ StageHeld kS := by
  rw [remote_check_ids] at h
  obtain ⟨hsame, hties⟩ := Properties.Composition.acceptTied _ _ _ _ _ h
  obtain ⟨hcheck, hstrays⟩ := checkSame_accept hsame
  obtain ⟨nI, nW, nS, kI, kW, kS, -, rfl, hidI, hidW, hpI, hpS, ⟨pI, cI, eI, hpI', hcI, heI, hchkI⟩, -, hS⟩ :=
    accepted_three hcheck
  have hS₀ := hS
  obtain ⟨pS, cS, eS, -, hcS, -, -⟩ := hS₀
  have hdS := strays_two (P₁ := NciCheck.Protocol.protocol) (P₂ := PouwCheck.Period.protocol)
    (P₃ := SampledProofs.OneStage.protocol) ne_policy_stage ne_period_stage hcI hcS hstrays
  have hp : (canonicalJson kI.params.toUTF8 >>= NciCheck.Protocol.paramsOf) = .ok pI := by rw [hpI]; exact hpI'
  refine ⟨kI, kW, kS, pI, cI, eI, rfl, digestOf_ok hcI,
    windows_tied hidI hidW hcI (hties _ (List.mem_singleton_self _)), (digestOf_ok hcS).trans (congrArg some hdS), hp, hcI,
    heI, hchkI, stage_held hpS hS⟩

theorem acceptInferenceOnly : AcceptInferenceOnly := by
  intro blobs ps c ev pub prof h
  obtain ⟨kI, kW, kS, pI, cI, eI, hev, hdI, hdW, hdS, hp, hc, he, ⟨pub', prof', hchk⟩, hstage⟩ := remote_accept h
  exact ⟨kI, kW, kS, NciCheck.Protocol.protocol.digest cI, hev, hdI, hdW, hdS, ⟨pI, cI, eI, hp, hc, he, policyHeld hchk⟩,
    hstage⟩

/-- **`AcceptInferenceOnly`'s nontrivial lemma**: its conclusion fails at no cases at all, `ev = []`, which a check that
accepts every input accepts: they are not the three subprotocols' cases. -/
theorem acceptInferenceOnly_nontrivial : ¬ InferenceOnlyHeld [] := by
  rintro ⟨kI, kW, kS, r, hev, -⟩
  cases hev

theorem acceptComplies : AcceptComplies := by
  intro blobs ps c ev pub prof h
  obtain ⟨kI, kW, kS, pI, cI, eI, hev, -, -, -, hp, hc, he, ⟨pub', prof', hchk⟩, -⟩ := remote_accept h
  obtain ⟨hn, out, hb, hcomp⟩ := policy_accept hchk
  obtain ⟨hd, -⟩ := NciCheck.Properties.SecurityProofs.BaselineSteps.evaluate_ok _ _ _ hb
  obtain ⟨P, hP, hname, -, ⟨j', i, d, n, U, hx, hpi, hL, hθ, -, hown, hmet⟩, -⟩ := policyHeld hchk
  refine ⟨kI, kW, kS, pI, cI, eI, P, hev, hp, hc, he, hP, hname, ⟨j', i, d, hx, hpi, hL, hθ, hown, hmet.2.1⟩, ?_⟩
  intro j' i d hx hi hL hN
  have hP' : Flock.Program.decode i.program.toUTF8 = .ok P := expands_program hx hi ▸ hP
  exact ⟨verifier_own hx hi, (NciCheck.Properties.SecurityProofs.DescWitnesses _ _ hd).2 j' hx i P d hi hP' hL hN hcomp⟩

/-- **`AcceptComplies`'s nontrivial lemma**, as `AcceptInferenceOnly`'s: its conclusion fails at `ev = []`, which a check
that accepts every input accepts. -/
theorem acceptComplies_nontrivial : ¬ CompliesHeld [] := by
  rintro ⟨kI, kW, kS, p, cr, e, P, hev, -⟩
  cases hev

end Properties.Remote.SecurityProofs
