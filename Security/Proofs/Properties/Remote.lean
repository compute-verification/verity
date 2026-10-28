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

/-- The ids of remote's subprotocols, and the subprotocols on the blobs `bs`, in their order. -/
abbrev ids : List String := ["verity/inference-only/v0", "verity/pouw/v1", "verity/sampled-proofs/one-stage/v2"]
abbrev parts (bs : Blobs) : List Protocol :=
  [NciCheck.Protocol.protocol, PouwCheck.protocol, SampledProofs.OneStageZk.protocol.onBlobs bs]

/-- Remote's acceptance, split into its three subprotocols' cases: each decodes to what its own check accepts, under
the parameters the composite names for it, and no record strays. -/
theorem remote_parts {blobs : Array ByteArray} {ps : List Named} {c : ByteArray} {ev : List Case} {pub prof : Keyed}
    (h : (Core.Remote.protocol.on blobs).check ps c ev = .accept pub prof) :
    ∃ (kI kW kS : Case) (nI nW nS : Named), ev = [kI, kW, kS] ∧ kI.params = nI.params ∧ kS.params = nS.params ∧
      Properties.Composition.DecodedAccepted NciCheck.Protocol.protocol nI.params.toUTF8 kI.commitment kI.evidence ∧
      Properties.Composition.DecodedAccepted PouwCheck.protocol nW.params.toUTF8 kW.commitment kW.evidence ∧
      Properties.Composition.DecodedAccepted (SampledProofs.OneStageZk.protocol.onBlobs (Blobs.ofArray blobs))
        nS.params.toUTF8 kS.commitment kS.evidence ∧
      strays ids (parts (Blobs.ofArray blobs)) [kI, kW, kS] = none := by
  rw [remote_check] at h
  obtain ⟨hcheck, hstrays⟩ := checkSame_accept h
  obtain ⟨⟨h1, h2, -, hpar⟩, hacc, -⟩ := Properties.Composition.acceptMeans _ _ _ _ _ _ _ hcheck
  have hlen : ev.length = 3 := by simpa using congrArg List.length h2
  have hlenp : ps.length = 3 := by simpa using congrArg List.length h1
  obtain ⟨kI, kW, kS, rfl⟩ : ∃ a b d, ev = [a, b, d] := by
    match ev, hlen with
    | [a, b, d], _ => exact ⟨a, b, d, rfl⟩
  obtain ⟨nI, nW, nS, rfl⟩ : ∃ a b d, ps = [a, b, d] := by
    match ps, hlenp with
    | [a, b, d], _ => exact ⟨a, b, d, rfl⟩
  have m1 : (NciCheck.Protocol.protocol, nI, kI) ∈
      Properties.Composition.triples (parts (Blobs.ofArray blobs)) [nI, nW, nS] [kI, kW, kS] := by
    simp [Properties.Composition.triples, parts]
  have m2 : (PouwCheck.protocol, nW, kW) ∈
      Properties.Composition.triples (parts (Blobs.ofArray blobs)) [nI, nW, nS] [kI, kW, kS] := by
    simp [Properties.Composition.triples, parts]
  have m3 : (SampledProofs.OneStageZk.protocol.onBlobs (Blobs.ofArray blobs), nS, kS) ∈
      Properties.Composition.triples (parts (Blobs.ofArray blobs)) [nI, nW, nS] [kI, kW, kS] := by
    simp [Properties.Composition.triples, parts]
  exact ⟨kI, kW, kS, nI, nW, nS, rfl, hpar _ m1, hpar _ m3, hacc _ m1, hacc _ m2, hacc _ m3, hstrays⟩

theorem onBlobs_params (P : BlobProtocol) (bs : Blobs) : (P.onBlobs bs).params = P.params := rfl
theorem onBlobs_commitment (P : BlobProtocol) (bs : Blobs) : (P.onBlobs bs).commitment = P.commitment := rfl
theorem onBlobs_digest (P : BlobProtocol) (bs : Blobs) : (P.onBlobs bs).digest = P.digest := rfl
theorem onBlobs_check (P : BlobProtocol) (bs : Blobs) (p : P.Params) (c : P.Commitment) (e : P.Evidence) :
    (P.onBlobs bs).check p c e = P.checkBlobs p c e bs := rfl

theorem policy_params : NciCheck.Protocol.protocol.params = NciCheck.Protocol.paramsOf := rfl
theorem policy_commitment : NciCheck.Protocol.protocol.commitment = NciCheck.Protocol.commitmentOf := rfl
theorem policy_evidence : NciCheck.Protocol.protocol.evidence = NciCheck.Protocol.evidenceOf := rfl
theorem policy_check : NciCheck.Protocol.protocol.check = NciCheck.Protocol.check := rfl
theorem stage_params : SampledProofs.OneStageZk.protocol.params = SampledProofs.OneStageZk.paramsOfJson := rfl
theorem stage_commitment : SampledProofs.OneStageZk.protocol.commitment = SampledProofs.OneStage.commitmentOf := rfl
theorem stage_checkBlobs : SampledProofs.OneStageZk.protocol.checkBlobs = SampledProofs.OneStageZk.checkBlobs := rfl

/-- The policy's case decodes, under the policy's own decoders, to what its check accepts. -/
theorem policy_decoded {a b e : ByteArray}
    (h : Properties.Composition.DecodedAccepted NciCheck.Protocol.protocol a b e) :
    ∃ (p : NciCheck.Protocol.Params) (c : NciCheck.Protocol.Commitment) (ev : NciCheck.Protocol.Evidence),
      (canonicalJson a >>= NciCheck.Protocol.paramsOf) = .ok p ∧ NciCheck.Protocol.commitmentOf b = .ok c ∧
      NciCheck.Protocol.evidenceOf e = .ok ev ∧ ∃ pub prof, NciCheck.Protocol.check p c ev = .accept pub prof := by
  obtain ⟨p, c, ev, hp, hc, he, pub, prof, hk⟩ := h
  rw [policy_params] at hp
  rw [policy_commitment] at hc
  rw [policy_evidence] at he
  rw [policy_check] at hk
  exact ⟨p, c, ev, hp, hc, he, pub, prof, hk⟩

/-- The one-stage audit's case decodes, under the audit's own decoders, to what its check on the blobs accepts. -/
theorem stage_decoded {bs : Blobs} {a b e : ByteArray}
    (h : Properties.Composition.DecodedAccepted (SampledProofs.OneStageZk.protocol.onBlobs bs) a b e) :
    ∃ (p : SampledProofs.OneStageZk.Params) (c : SampledProofs.OneStage.Commitment)
      (ev : SampledProofs.OneStageZk.Evidence),
      (canonicalJson a >>= SampledProofs.OneStageZk.paramsOfJson) = .ok p ∧
      SampledProofs.OneStage.commitmentOf b = .ok c ∧
      ∃ pub prof, SampledProofs.OneStageZk.checkBlobs p c ev bs = .accept pub prof := by
  obtain ⟨p, c, ev, hp, hc, -, pub, prof, hk⟩ := h
  rw [onBlobs_params, stage_params] at hp
  rw [onBlobs_commitment, stage_commitment] at hc
  rw [onBlobs_check, stage_checkBlobs] at hk
  exact ⟨p, c, ev, hp, hc, pub, prof, hk⟩

theorem policy_digest (c : NciCheck.Protocol.Commitment) : NciCheck.Protocol.protocol.digest c = c.digest := rfl
theorem stage_digest (bs : Blobs) (c : SampledProofs.OneStage.Commitment) :
    (SampledProofs.OneStageZk.protocol.onBlobs bs).digest c = c.digest := rfl

theorem committed_policy {bs : Blobs} {kI kW kS : Case} {cI : NciCheck.Protocol.Commitment}
    (hc : NciCheck.Protocol.protocol.commitment kI.commitment = .ok cI) :
    committedBy "verity/inference-only/v0" (parts bs) [kI, kW, kS] = some cI.digest := by
  simp [committedBy, parts, id_policy, hc, Except.toOption, policy_digest]

theorem committed_pouw {bs : Blobs} {kI kW kS : Case} {cW : PouwCheck.protocol.Commitment}
    (hc : PouwCheck.protocol.commitment kW.commitment = .ok cW) :
    committedBy "verity/pouw/v1" (parts bs) [kI, kW, kS] = some (PouwCheck.protocol.digest cW) := by
  simp [committedBy, parts, id_policy, id_pouw, hc, Except.toOption]

theorem committed_stage {bs : Blobs} {kI kW kS : Case} {cS : SampledProofs.OneStage.Commitment}
    (hc : SampledProofs.OneStage.commitmentOf kS.commitment = .ok cS) :
    committedBy "verity/sampled-proofs/one-stage/v2" (parts bs) [kI, kW, kS] = some cS.digest := by
  have hc' : (SampledProofs.OneStageZk.protocol.onBlobs bs).commitment kS.commitment = .ok cS := by
    rw [onBlobs_commitment, stage_commitment]; exact hc
  simp [committedBy, parts, id_policy, id_pouw, id_oneStage, hc', Except.toOption, stage_digest]

/-- With no stray, the three records' digests are the policy's record's. -/
theorem tie {bs : Blobs} {kI kW kS : Case} {cI : NciCheck.Protocol.Commitment} {cW : PouwCheck.protocol.Commitment}
    {cS : SampledProofs.OneStage.Commitment}
    (hcI : NciCheck.Protocol.commitmentOf kI.commitment = .ok cI)
    (hcW : PouwCheck.protocol.commitment kW.commitment = .ok cW)
    (hcS : SampledProofs.OneStage.commitmentOf kS.commitment = .ok cS)
    (hs : strays ids (parts bs) [kI, kW, kS] = none) :
    Properties.Composition.digestOf NciCheck.Protocol.protocol kI.commitment = some cI.digest ∧
      Properties.Composition.digestOf PouwCheck.protocol kW.commitment = some cI.digest ∧
      Properties.Composition.digestOf SampledProofs.OneStageZk.protocol.toProtocol kS.commitment = some cI.digest := by
  have hcI' : NciCheck.Protocol.protocol.commitment kI.commitment = .ok cI := by rw [policy_commitment]; exact hcI
  have hm : ids.mapM (fun s => (committedBy s (parts bs) [kI, kW, kS]).map (s, ·)) =
      some [("verity/inference-only/v0", cI.digest), ("verity/pouw/v1", PouwCheck.protocol.digest cW),
        ("verity/sampled-proofs/one-stage/v2", cS.digest)] := by
    simp [ids, committed_policy hcI', committed_pouw hcW, committed_stage hcS]
  have htie := strays_none hm hs
  have hdW : PouwCheck.protocol.digest cW = cI.digest := htie _ (by simp)
  have hdS : cS.digest = cI.digest := htie _ (by simp)
  have hcS' : SampledProofs.OneStageZk.protocol.toProtocol.commitment kS.commitment = .ok cS := by
    rw [← stage_commitment] at hcS; exact hcS
  refine ⟨?_, ?_, ?_⟩
  · simp [Properties.Composition.digestOf, hcI', Except.toOption, policy_digest]
  · simp [Properties.Composition.digestOf, hcW, Except.toOption, hdW]
  · have e : SampledProofs.OneStageZk.protocol.toProtocol.digest cS = cS.digest := rfl
    simp [Properties.Composition.digestOf, hcS', Except.toOption, e, hdS]

/-- What remote's acceptance gives: the three cases in order, one digest for their records, and the policy's case
decoding to what the policy's check accepts. -/
theorem remote_accept {blobs : Array ByteArray} {ps : List Named} {c : ByteArray} {ev : List Case} {pub prof : Keyed}
    (h : (Core.Remote.protocol.on blobs).check ps c ev = .accept pub prof) :
    ∃ (kI kW kS : Case) (pI : NciCheck.Protocol.Params) (cI : NciCheck.Protocol.Commitment)
      (eI : NciCheck.Protocol.Evidence), ev = [kI, kW, kS] ∧
      Properties.Composition.digestOf NciCheck.Protocol.protocol kI.commitment = some cI.digest ∧
      Properties.Composition.digestOf PouwCheck.protocol kW.commitment = some cI.digest ∧
      Properties.Composition.digestOf SampledProofs.OneStageZk.protocol.toProtocol kS.commitment = some cI.digest ∧
      (canonicalJson kI.params.toUTF8 >>= NciCheck.Protocol.paramsOf) = .ok pI ∧
      NciCheck.Protocol.commitmentOf kI.commitment = .ok cI ∧ NciCheck.Protocol.evidenceOf kI.evidence = .ok eI ∧
      (∃ pub' prof', NciCheck.Protocol.check pI cI eI = .accept pub' prof') ∧ StageHeld kS := by
  obtain ⟨kI, kW, kS, nI, nW, nS, rfl, hpI, hpS, hI, hW, hS, hs⟩ := remote_parts h
  obtain ⟨pI, cI, eI, hp, hcI, heI, hk⟩ := policy_decoded hI
  obtain ⟨_, cW, _, _, hcW, _⟩ := hW
  obtain ⟨pS, cS, _, hpS', hcS, _, _, hkS⟩ := stage_decoded hS
  obtain ⟨d1, d2, d3⟩ := tie hcI hcW hcS hs
  refine ⟨kI, kW, kS, pI, cI, eI, rfl, d1, d2, d3, by rw [hpI]; exact hp, hcI, heI, hk, ?_⟩
  exact ⟨pS, cS, by rw [hpS]; exact hpS', hcS, stage_registered hkS⟩

theorem acceptInferenceOnly : AcceptInferenceOnly := by
  intro blobs ps c ev pub prof h
  obtain ⟨kI, kW, kS, pI, cI, eI, hev, hdI, hdW, hdS, hp, hc, he, ⟨pub', prof', hchk⟩, hstage⟩ := remote_accept h
  exact ⟨kI, kW, kS, cI.digest, hev, hdI, hdW, hdS, ⟨pI, cI, eI, hp, hc, he, policyHeld hchk⟩, hstage⟩

theorem acceptInferenceOnly_by :
    AcceptInferenceOnly ↔ ∀ blobs, AcceptInferenceOnlyBy (Core.Remote.protocol.on blobs).check := Iff.rfl

/-- **No check that accepts every input has `AcceptInferenceOnly`'s property** (its nontrivial lemma): such a check
accepts no cases at all, `ev = []`, which are not the three subprotocols' cases. -/
theorem acceptInferenceOnly_nontrivial : ∀ v, AcceptsAll v → ¬ AcceptInferenceOnlyBy v := by
  intro v hall h
  obtain ⟨pub, prof, hv⟩ := hall [] ByteArray.empty []
  obtain ⟨kI, kW, kS, r, hev, -⟩ := h [] ByteArray.empty [] pub prof hv
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

end Properties.Remote.SecurityProofs
