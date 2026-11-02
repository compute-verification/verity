import Proofs.Properties.Remote
import Proofs.Flock.Claims.Complete.Json.Hex

/-!
# Remote's tie on an honest run: one record, one digest

The honest lemma of `AcceptInferenceOnly`'s tie: a run commits its three subprotocols to its one registration record,
and whenever the three subprotocols' commitment decoders all accept that record's bytes, they decode them to one digest
(`one_digest`), so no record strays and the three digests are the policy's record's (`acceptInferenceOnly_honest`).
The record may name any registered circuit, a tiny one among them: the commitment decoders read only its `format`.

The lemma takes the decoders' acceptance of the bytes as its hypotheses rather than exhibiting concrete bytes: the
decoders read their bytes through `Lean.Json.parse`, which is `partial`, so no proof evaluates them on a literal.
-/

namespace Properties.Remote.SecurityProofs

open Core Core.Composition
open Lean (Json)

private theorem toUTF8_of_fromUTF8? {b : ByteArray} {s : String} (h : String.fromUTF8? b = some s) :
    s.toUTF8 = b := by
  unfold String.fromUTF8? at h
  split at h
  · cases h; rfl
  · cases h

/-- Canonical bytes are their record's canonical text. -/
theorem canonical_text {b : ByteArray} {j : Json} (h : canonicalJson b = .ok j) :
    ∃ s, Flock.canon j = .ok s ∧ s.toUTF8 = b := by
  unfold canonicalJson at h
  rcases hu : String.fromUTF8? b with _ | s
  · simp [hu, throw, throwThe, MonadExceptOf.throw] at h
    split at h <;> simp [Bind.bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
  · simp only [hu] at h
    rcases hp : Json.parse s with e | j0
    · simp [hp, throw, throwThe, MonadExceptOf.throw, Bind.bind, Except.bind] at h
      split at h <;> simp [Bind.bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
    · rcases hc : Flock.canon j0 with e | t
      · simp [hp, hc, Bind.bind, Except.bind, pure, Except.pure] at h
        split at h <;> simp [Bind.bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
      · by_cases ht : t = s
        · subst ht
          simp [hp, hc, Bind.bind, Except.bind, pure, Except.pure] at h
          split at h
          · cases h
            exact ⟨_, hc, toUTF8_of_fromUTF8? hu⟩
          · simp [Bind.bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
        · simp [hp, hc, Bind.bind, Except.bind, pure, Except.pure, ht, throw, throwThe, MonadExceptOf.throw] at h
          split at h <;> simp [Bind.bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h

/-- A record's own tag (`Flock.WindowRecord.tagOf`) is the one the policy's digest takes: both read its `format` field
as a string. -/
theorem tagOf_eq (j : Json) : Flock.WindowRecord.tagOf j =
    if NciCheck.Baseline.strAt j "format" == some Flock.WindowRecord.FORMAT_V2 then Flock.WindowRecord.FORMAT_V2
    else Flock.WindowRecord.FORMAT := by
  have hv : (j.getObjValAs? String "format").toOption = NciCheck.Baseline.strAt j "format" := by
    show (Lean.fromJson? (j.getObjValD "format") : Except String String).toOption =
      NciCheck.Baseline.strOf? (j.getObjValD "format")
    cases j.getObjValD "format" <;> rfl
  unfold Flock.WindowRecord.tagOf
  rw [hv]

theorem push_zero (x : ByteArray) : x.push 0 = x ++ ByteArray.mk #[0] := by
  apply ByteArray.ext
  simp [ByteArray.data_push, ByteArray.data_append]

/-- The policy's digest is the window record's (`Flock.WindowRecord.digestHex`): `NciCheck.Baseline.digest` is that
function, or else tags the record by its `format` as `Flock.WindowRecord.tagOf` does and hashes its canonical text as
`Flock.WindowRecord.digest` does, and either form is proved here. -/
theorem baseline_digest (rec : Json) : NciCheck.Baseline.digest rec = Flock.WindowRecord.digestHex rec := by
  first
    | rfl
    | (unfold NciCheck.Baseline.digest Flock.WindowRecord.digestHex Flock.WindowRecord.digest
       rw [tagOf_eq]
       cases Flock.canon rec with
       | error e => rfl
       | ok s => simp only [Bind.bind, Except.bind, pure, Except.pure, Flock.WindowRecord.digestOf, push_zero] <;> rfl)

/-- The policy's commitment: the record its bytes are, and its digest, the window record's digest of that record. -/
theorem policy_of {b : ByteArray} {c : NciCheck.Protocol.Commitment} (h : NciCheck.Protocol.commitmentOf b = .ok c) :
    canonicalJson b = .ok c.record ∧ Flock.WindowRecord.digest c.record = .ok c.digest := by
  unfold NciCheck.Protocol.commitmentOf at h
  simp only [baseline_digest, Flock.WindowRecord.digestHex] at h
  rcases hj : canonicalJson b with e | j
  · simp [hj, Bind.bind, Except.bind] at h
  rcases hd : Flock.WindowRecord.digest j with e | d
  · cases j <;> simp [hj, hd, Bind.bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  cases j <;> simp [hj, hd, Bind.bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw,
    FlockComplete.JsonRT.unhex_hexOf] at h
  subst h
  exact ⟨rfl, hd⟩

/-- The one-stage audit's commitment: the record its bytes are, and `Registration.digest` of it. -/
theorem stage_of {b : ByteArray} {c : SampledProofs.OneStage.Commitment}
    (h : SampledProofs.OneStage.commitmentOf b = .ok c) :
    canonicalJson b = .ok c.record ∧ SampledProofs.Registration.digest c.record = .ok c.digest := by
  unfold SampledProofs.OneStage.commitmentOf at h
  rcases hj : canonicalJson b with e | j
  · simp [hj, Bind.bind, Except.bind] at h
  rcases hg : SampledProofs.Registration.digest j with e | d
  · cases j <;> simp [hj, hg, Bind.bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  cases j <;> simp [hj, hg, Bind.bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  subst h
  exact ⟨rfl, hg⟩

/-- PoUW's commitment: the record its bytes are, a format among PoUW's, and the bytes themselves. -/
theorem pouw_of {b : ByteArray} {c : PouwCheck.Registration} (h : PouwCheck.registrationOf b = .ok c) :
    ∃ j, canonicalJson b = .ok j ∧
      (PouwCheck.str? (PouwCheck.get j "format")).filter (PouwCheck.formats.contains ·) = some c.format ∧
      c.bytes = b := by
  unfold PouwCheck.registrationOf at h
  rcases hj : canonicalJson b with e | j
  · simp [hj, Bind.bind, Except.bind] at h
  simp only [hj, Bind.bind, Except.bind] at h
  rcases hf : (PouwCheck.str? (PouwCheck.get j "format")).filter (PouwCheck.formats.contains ·) with _ | f
  · rw [hf] at h
    simp [throw, throwThe, MonadExceptOf.throw] at h
  rw [hf] at h
  simp only [pure, Except.pure] at h
  repeat' (first | (split at h) | (cases h; exact ⟨j, rfl, hf, rfl⟩) | cases h)

theorem pouw_get (j : Json) (k : String) : PouwCheck.get j k = NciCheck.Baseline.getD j k := by
  cases j <;> rfl

/-- **One record, one digest**: bytes that the three subprotocols' commitment decoders all accept decode to one digest
under each of them. -/
theorem one_digest {b : ByteArray} {cI : NciCheck.Protocol.Commitment} {cW : PouwCheck.Registration}
    {cS : SampledProofs.OneStage.Commitment} (hI : NciCheck.Protocol.commitmentOf b = .ok cI)
    (hW : PouwCheck.registrationOf b = .ok cW) (hS : SampledProofs.OneStage.commitmentOf b = .ok cS) :
    cW.digest = cI.digest ∧ cS.digest = cI.digest := by
  obtain ⟨hjI, hdI⟩ := policy_of hI
  obtain ⟨j, hjW, hf, hbytes⟩ := pouw_of hW
  obtain ⟨hjS, hdS⟩ := stage_of hS
  have e1 : cI.record = j := by rw [hjI] at hjW; exact Except.ok.inj hjW
  have e2 : cS.record = j := by rw [hjS] at hjW; exact Except.ok.inj hjW
  rw [e1] at hdI
  rw [e2] at hdS
  refine ⟨?_, Except.ok.inj (hdS.symm.trans hdI)⟩
  obtain ⟨s, hs, hsb⟩ := canonical_text hjW
  unfold Flock.WindowRecord.digest at hdI
  simp only [hs, Bind.bind, Except.bind, pure, Except.pure] at hdI
  have hdI' := Except.ok.inj hdI
  simp only [Option.filter_eq_some_iff] at hf
  obtain ⟨hstr, hmem⟩ := hf
  have hget : NciCheck.Baseline.getD j "format" = .str cW.format := by
    rw [← pouw_get]
    revert hstr
    cases PouwCheck.get j "format" <;> simp [PouwCheck.str?]
  have hT : Flock.WindowRecord.tagOf j = cW.format := by
    have hv : j.getObjValAs? String "format" = .ok cW.format := by
      unfold Lean.Json.getObjValAs?
      rw [show j.getObjValD "format" = .str cW.format from hget]
      rfl
    have hfm : cW.format = Flock.WindowRecord.FORMAT ∨ cW.format = Flock.WindowRecord.FORMAT_V2 := by
      simpa [PouwCheck.formats, Flock.WindowRecord.FORMAT, Flock.WindowRecord.FORMAT_V2] using hmem
    unfold Flock.WindowRecord.tagOf
    rw [hv]
    rcases hfm with h | h <;> rw [h] <;> decide +kernel
  unfold PouwCheck.Registration.digest
  rw [← hdI', hbytes, ← hsb, hT]

/-- **Remote's tie holds on an honest run** (the honest lemma of `AcceptInferenceOnly`'s tie): three cases committing
to one record's bytes, which the three subprotocols' commitment decoders accept, leave no stray among the components
remote's check compares (`remote_check_ids`), and their three digests are the policy's record's. -/
theorem acceptInferenceOnly_honest {kI kW kS : Case} {cI : NciCheck.Protocol.Commitment}
    {cW : PouwCheck.Registration} {cS : SampledProofs.OneStage.Commitment}
    (hW' : kW.commitment = kI.commitment) (hS' : kS.commitment = kI.commitment)
    (hI : NciCheck.Protocol.commitmentOf kI.commitment = .ok cI)
    (hW : PouwCheck.registrationOf kI.commitment = .ok cW)
    (hS : SampledProofs.OneStage.commitmentOf kI.commitment = .ok cS) :
    strays [NciCheck.Protocol.protocol.id, PouwCheck.protocol.id, SampledProofs.OneStage.protocol.id]
        [NciCheck.Protocol.protocol, PouwCheck.protocol, SampledProofs.OneStage.protocol] [kI, kW, kS] = none ∧
      Properties.Composition.digestOf NciCheck.Protocol.protocol kI.commitment =
        some (NciCheck.Protocol.protocol.digest cI) ∧
      Properties.Composition.digestOf PouwCheck.protocol kW.commitment =
        some (NciCheck.Protocol.protocol.digest cI) ∧
      Properties.Composition.digestOf SampledProofs.OneStage.protocol kS.commitment =
        some (NciCheck.Protocol.protocol.digest cI) := by
  obtain ⟨dW, dS⟩ := one_digest hI hW hS
  have hcI : NciCheck.Protocol.protocol.commitment kI.commitment = .ok cI := hI
  have hcW : PouwCheck.protocol.commitment kW.commitment = .ok cW := by rw [hW']; exact hW
  have hcS : SampledProofs.OneStage.protocol.commitment kS.commitment = .ok cS := by rw [hS']; exact hS
  have dW' : PouwCheck.protocol.digest cW = NciCheck.Protocol.protocol.digest cI := dW
  have dS' : SampledProofs.OneStage.protocol.digest cS = NciCheck.Protocol.protocol.digest cI := dS
  exact ⟨strays_three_none ne_policy_pouw ne_policy_stage ne_pouw_stage hcI hcW hcS dW' dS',
    digestOf_ok hcI, (digestOf_ok hcW).trans (congrArg some dW'), (digestOf_ok hcS).trans (congrArg some dS')⟩

end Properties.Remote.SecurityProofs
