import Properties.Remote
import Proofs.Properties.Composition
import Proofs.Properties.Nci.DescVerdict
import Proofs.Properties.Nci.Baseline
import Proofs.Properties.Nci.BaselineSteps
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
`Verity.Circuit.expand_boolean` give the padded `(c, k)`-partitions.
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

/-- What a verdict of compliance on the policy's instance gives, on each of its expansions. -/
theorem policy_met {p : NciCheck.Protocol.Params} {e : NciCheck.Protocol.Evidence} {out : Json}
    (hd : NciCheck.Desc.evaluate (instanceOf p e) = .ok out) (hc : out.getObjVal? "complies" = .ok (.bool true))
    {P : Prog} (hP : Flock.Program.decode p.program.toUTF8 = .ok P) :
    ∀ (j' : Json) (i : NciCheck.Instance) (d : Declaration) (n : ℕ) (U : Partition d.D n),
      Expands (instanceOf p e) j' → Parses j' i → LaysOut P d.D → Names i.theta d.θ → Divides i.owner U →
        i.program = p.program ∧ PolicyMet d U i.gamma ((i.num : ℝ) / i.den) := by
  intro j' i d n U hx hi hL hN hD
  have hprog := expands_program hx hi
  have hP' : Flock.Program.decode i.program.toUTF8 = .ok P := hprog ▸ hP
  have hR := (NciCheck.Properties.SecurityProofs.DescReports _ _ hd).2 j' hx i P d n U hi hP' hL hN hD
  have hmeets : ∀ u, Meets d U i.gamma ((i.num : ℝ) / i.den) u := hR.2.2.1.1 hc
  have hcomp : Nci.Complies i.gamma ((i.num : ℝ) / i.den) d := ⟨n, U, hmeets⟩
  have hα : (0 : ℝ) ≤ (i.num : ℝ) / i.den := div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
  refine ⟨hprog, hmeets, hcomp, fun hB => Nci.SecurityProofs.CompliesCK _ _ d hB hcomp, ?_⟩
  intro W hfan hw hW
  have hx' := Nci.SecurityProofs.ExpandComplies i.gamma W _ d hα hw hW hcomp
  exact Nci.SecurityProofs.CompliesCK _ _ d.expand (d.D.expand_boolean hfan) hx'

/-- What the policy's acceptance establishes about its decoded case. -/
theorem policyHeld {p : NciCheck.Protocol.Params} {c : NciCheck.Protocol.Commitment}
    {e : NciCheck.Protocol.Evidence} {pub : Unit} {prof : Json}
    (h : NciCheck.Protocol.check p c e = .accept pub prof) : PolicyHeld p c e := by
  obtain ⟨hn, out, hb, hc⟩ := policy_accept h
  obtain ⟨hd, -⟩ := NciCheck.Properties.SecurityProofs.BaselineSteps.evaluate_ok _ _ _ hb
  obtain ⟨⟨j', hx⟩, -⟩ := NciCheck.Properties.SecurityProofs.DescReports _ _ hd
  obtain ⟨i0, P, h0, hP0, -⟩ := hx
  have hprog : i0.program = p.program := by
    have a : (instanceOf p e).getObjVal? "program" = .ok (.str i0.program) := h0.1
    rw [instance_program] at a
    exact (Json.str.inj (Except.ok.inj a)).symm
  have hP : Flock.Program.decode p.program.toUTF8 = .ok P := hprog ▸ hP0
  exact ⟨P, hP, names_program hn hP, NciCheck.Properties.SecurityProofs.BaselineProvenance _ _ _ hb,
    (NciCheck.Properties.SecurityProofs.DescReports _ _ hd).1, policy_met hd hc hP⟩

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
      ∃ pub' prof', NciCheck.Protocol.check pI cI eI = .accept pub' prof' := by
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
  obtain ⟨pS, cS, eS, -, hcS, -, -⟩ := hS
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
  refine ⟨kI, kW, kS, pI, cI, eI, rfl, ?_, ?_, ?_, hp, hcI', heI', pubI, profI, hchkI'⟩
  · simp [Properties.Composition.digestOf, hcI', Except.toOption]
  · simp [Properties.Composition.digestOf, hcW', Except.toOption, hdW]
  · simp [Properties.Composition.digestOf, hcS'', Except.toOption, hdS']

theorem acceptInferenceOnly : AcceptInferenceOnly := by
  intro blobs ps c ev pub prof h
  obtain ⟨kI, kW, kS, pI, cI, eI, hev, hdI, hdW, hdS, hp, hc, he, pub', prof', hchk⟩ := remote_accept h
  exact ⟨kI, kW, kS, NciCheck.Protocol.protocol.digest cI, hev, hdI, hdW, hdS, pI, cI, eI, hp, hc, he, policyHeld hchk⟩

theorem acceptComplies : AcceptComplies := by
  intro blobs ps c ev pub prof h
  obtain ⟨kI, kW, kS, pI, cI, eI, hev, -, -, -, hp, hc, he, pub', prof', hchk⟩ := remote_accept h
  obtain ⟨hn, out, hb, hcomp⟩ := policy_accept hchk
  obtain ⟨hd, -⟩ := NciCheck.Properties.SecurityProofs.BaselineSteps.evaluate_ok _ _ _ hb
  obtain ⟨P, hP, hname, -, -, -⟩ := policyHeld hchk
  refine ⟨kI, kW, kS, pI, cI, eI, P, hev, hp, hc, he, hP, hname, ?_⟩
  intro j' i d hx hi hL hN
  have hP' : Flock.Program.decode i.program.toUTF8 = .ok P := expands_program hx hi ▸ hP
  exact (NciCheck.Properties.SecurityProofs.DescWitnesses _ _ hd).2 j' hx i P d hi hP' hL hN hcomp

end Properties.Remote.SecurityProofs
