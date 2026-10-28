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

Remote's check is `Core.Composition.checkSame` over its components on the caller's blobs (`remote_check`), so an
acceptance is `Core.Composition.check`'s (`checkSame_accept`), which `Properties.Composition.acceptMeans` reduces to each
component accepting its own case, with no component's record other than the first's (`strays_none`). The policy's
acceptance is `NciCheck.Baseline.evaluate`'s with the record naming the circuit (`policy_accept`), which is
`NciCheck.Desc.evaluate`'s and θ's provenance (`BaselineSteps.evaluate_ok`), whose verdict `DescReports`,
`DescWitnesses` and `BaselineProvenance` read; `Nci.SecurityProofs.CompliesCK`, `ExpandComplies` and
`Verity.Circuit.expand_boolean` give the padded `(c, k)`-partitions. The parse, declaration and division an acceptance
exhibits come from `DescAgrees` (the verdict is the gate-by-gate check's on an expansion) and `Decodes` (that check's
verdicts are about a parse, a layout, a baseline and a division).

The policy's existence clause is neither empty nor trivial: `policyMet_honest` exhibits a declaration and a division into
nonempty inference blocks that meet the policy, and `policyMet_nontrivial` one the policy refuses at every `α`.
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

/-- Remote's check on the caller's blobs is `checkSame` over its components on them. -/
theorem remote_check (blobs : Array ByteArray) (ps : List Named) (c : ByteArray) (ev : List Case) :
    (Core.Remote.protocol.on blobs).check ps c ev =
      checkSame "verity/remote/v0" ["verity/inference-only/v0", "verity/pouw/v1", "verity/sampled-proofs/one-stage/v2"]
        [NciCheck.Protocol.protocol, PouwCheck.protocol, SampledProofs.OneStageZk.protocol.onBlobs (Blobs.ofArray blobs)]
        ps c ev := rfl

theorem id_policy : NciCheck.Protocol.protocol.id = "verity/inference-only/v0" := rfl
theorem id_pouw : PouwCheck.protocol.id = "verity/pouw/v1" := rfl
theorem id_oneStage (bs : Blobs) :
    (SampledProofs.OneStageZk.protocol.onBlobs bs).id = "verity/sampled-proofs/one-stage/v2" := rfl

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
theorem stage_registered {p : SampledProofs.OneStageZk.Params} {c : SampledProofs.OneStage.Commitment}
    {e : SampledProofs.OneStageZk.Evidence} {bs : Blobs} {pub : SampledProofs.OneStage.Public} {prof : Integrity}
    (h : SampledProofs.OneStageZk.checkBlobs p c e bs = .accept pub prof) :
    Properties.SampledProofs.R2 c.record p.base.verifier := by
  unfold SampledProofs.OneStageZk.checkBlobs at h
  split at h
  · cases h
  · split at h
    · rename_i pub' prof' ho
      obtain ⟨law, d, n, hreg, -⟩ := Properties.SampledProofs.acceptMeans _ _ _ _ _ ho
      exact hreg.2.1
    · simp_all

/-- What remote's acceptance gives: the three cases in order, one digest for their records, and the policy's case
decoding to what the policy's check accepts. -/
theorem remote_accept {blobs : Array ByteArray} {ps : List Named} {c : ByteArray} {ev : List Case} {pub prof : Keyed}
    (h : (Core.Remote.protocol.on blobs).check ps c ev = .accept pub prof) :
    ∃ (kI kW kS : Case) (pI : NciCheck.Protocol.Params) (cI : NciCheck.Protocol.Commitment)
      (eI : NciCheck.Protocol.Evidence), ev = [kI, kW, kS] ∧
      Properties.Composition.digestOf NciCheck.Protocol.protocol kI.commitment = some (NciCheck.Protocol.protocol.digest cI) ∧
      Properties.Composition.digestOf PouwCheck.protocol kW.commitment = some (NciCheck.Protocol.protocol.digest cI) ∧
      Properties.Composition.digestOf SampledProofs.OneStageZk.protocol.toProtocol kS.commitment = some (NciCheck.Protocol.protocol.digest cI) ∧
      (canonicalJson kI.params.toUTF8 >>= NciCheck.Protocol.paramsOf) = .ok pI ∧
      NciCheck.Protocol.commitmentOf kI.commitment = .ok cI ∧ NciCheck.Protocol.evidenceOf kI.evidence = .ok eI ∧
      (∃ pub' prof', NciCheck.Protocol.check pI cI eI = .accept pub' prof') ∧ StageHeld kS := by
  rw [remote_check] at h
  obtain ⟨hcheck, hstrays⟩ := checkSame_accept h
  obtain ⟨⟨h1, h2, -, hpar⟩, hacc, -⟩ := Properties.Composition.acceptMeans _ _ _ _ _ _ _ hcheck
  have hlen : ev.length = 3 := by simpa using congrArg List.length h2
  have hlenp : ps.length = 3 := by simpa using congrArg List.length h1
  rcases ev with _ | ⟨kI, _ | ⟨kW, _ | ⟨kS, _ | ⟨_, _⟩⟩⟩⟩ <;> simp at hlen
  rcases ps with _ | ⟨nI, _ | ⟨nW, _ | ⟨nS, _ | ⟨_, _⟩⟩⟩⟩ <;> simp at hlenp
  have hI := hacc (NciCheck.Protocol.protocol, nI, kI) (by simp [Properties.Composition.triples])
  have hW := hacc (PouwCheck.protocol, nW, kW) (by simp [Properties.Composition.triples])
  have hS := hacc (SampledProofs.OneStageZk.protocol.onBlobs (Blobs.ofArray blobs), nS, kS)
    (by simp [Properties.Composition.triples])
  have hpI : kI.params = nI.params := hpar (NciCheck.Protocol.protocol, nI, kI) (by simp [Properties.Composition.triples])
  obtain ⟨pI, cI, eI, hpI', hcI, heI, pubI, profI, hchkI⟩ := hI
  obtain ⟨pW, cW, eW, -, hcW, -, -⟩ := hW
  have hpS : kS.params = nS.params := hpar (SampledProofs.OneStageZk.protocol.onBlobs (Blobs.ofArray blobs), nS, kS)
    (by simp [Properties.Composition.triples])
  obtain ⟨pS, cS, eS, hpS', hcS, -, pubS, profS, hchkS⟩ := hS
  have hpS'' : (canonicalJson nS.params.toUTF8 >>= SampledProofs.OneStageZk.paramsOfJson) = .ok pS := hpS'
  have hcS₀ : SampledProofs.OneStage.commitmentOf kS.commitment = .ok cS := hcS
  have hchkS' : SampledProofs.OneStageZk.checkBlobs pS cS eS (Blobs.ofArray blobs) = .accept pubS profS := hchkS
  have hstage : StageHeld kS :=
    ⟨pS, cS, by rw [hpS]; exact hpS'', hcS₀, stage_registered hchkS'⟩
  have hcI' : NciCheck.Protocol.protocol.commitment kI.commitment = .ok cI := hcI
  have hcW' : PouwCheck.protocol.commitment kW.commitment = .ok cW := hcW
  have hcS' : (SampledProofs.OneStageZk.protocol.onBlobs (Blobs.ofArray blobs)).commitment kS.commitment = .ok cS := hcS
  have hpI'' : (canonicalJson nI.params.toUTF8 >>= NciCheck.Protocol.paramsOf) = .ok pI := hpI'
  have heI' : NciCheck.Protocol.evidenceOf kI.evidence = .ok eI := heI
  have hchkI' : NciCheck.Protocol.check pI cI eI = .accept pubI profI := hchkI
  have hm : ["verity/inference-only/v0", "verity/pouw/v1", "verity/sampled-proofs/one-stage/v2"].mapM
      (fun s => (committedBy s [NciCheck.Protocol.protocol, PouwCheck.protocol,
        SampledProofs.OneStageZk.protocol.onBlobs (Blobs.ofArray blobs)] [kI, kW, kS]).map (s, ·)) =
      some [("verity/inference-only/v0", NciCheck.Protocol.protocol.digest cI),
        ("verity/pouw/v1", PouwCheck.protocol.digest cW),
        ("verity/sampled-proofs/one-stage/v2",
          (SampledProofs.OneStageZk.protocol.onBlobs (Blobs.ofArray blobs)).digest cS)] := by
    simp [committedBy, id_policy, id_pouw, id_oneStage, hcI', hcW', hcS', Except.toOption]
  have htie := strays_none hm hstrays
  have hdW : PouwCheck.protocol.digest cW = NciCheck.Protocol.protocol.digest cI := htie _ (by simp)
  have hdS : (SampledProofs.OneStageZk.protocol.onBlobs (Blobs.ofArray blobs)).digest cS =
      NciCheck.Protocol.protocol.digest cI := htie _ (by simp)
  have hcS'' : SampledProofs.OneStageZk.protocol.toProtocol.commitment kS.commitment = .ok cS := hcS
  have hdS' : SampledProofs.OneStageZk.protocol.toProtocol.digest cS = NciCheck.Protocol.protocol.digest cI := hdS
  have hp : (canonicalJson kI.params.toUTF8 >>= NciCheck.Protocol.paramsOf) = .ok pI := by rw [hpI]; exact hpI''
  refine ⟨kI, kW, kS, pI, cI, eI, rfl, ?_, ?_, ?_, hp, hcI', heI', ⟨pubI, profI, hchkI'⟩, hstage⟩
  · simp [Properties.Composition.digestOf, hcI', Except.toOption]
  · simp [Properties.Composition.digestOf, hcW', Except.toOption, hdW]
  · simp [Properties.Composition.digestOf, hcS'', Except.toOption, hdS']

theorem acceptInferenceOnly : AcceptInferenceOnly := by
  intro blobs ps c ev pub prof h
  obtain ⟨kI, kW, kS, pI, cI, eI, hev, hdI, hdW, hdS, hp, hc, he, ⟨pub', prof', hchk⟩, hstage⟩ := remote_accept h
  exact ⟨kI, kW, kS, NciCheck.Protocol.protocol.digest cI, hev, hdI, hdW, hdS, ⟨pI, cI, eI, hp, hc, he, policyHeld hchk⟩,
    hstage⟩

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

end Properties.Remote.SecurityProofs
