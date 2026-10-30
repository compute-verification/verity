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

private theorem canon_str (s : String) : Flock.canon (.str s) = pure (Flock.canonStr s) := by
  rw [Flock.canon]

/-- Canonical bytes are their record's canonical text. -/
theorem canonical_text {b : ByteArray} {j : Json} (h : canonicalJson b = .ok j) :
    ∃ s, Flock.canon j = .ok s ∧ s.toUTF8 = b := by
  unfold canonicalJson at h
  rcases hu : String.fromUTF8? b with _ | s
  · simp [hu, throw, throwThe, MonadExceptOf.throw] at h
  · simp only [hu] at h
    rcases hp : Json.parse s with e | j0
    · simp [hp, throw, throwThe, MonadExceptOf.throw, Bind.bind, Except.bind] at h
    · rcases hc : Flock.canon j0 with e | t
      · simp [hp, hc, Bind.bind, Except.bind, pure, Except.pure] at h
      · by_cases ht : t = s
        · subst ht
          simp [hp, hc, Bind.bind, Except.bind, pure, Except.pure] at h
          subst h
          exact ⟨_, hc, toUTF8_of_fromUTF8? hu⟩
        · simp [hp, hc, Bind.bind, Except.bind, pure, Except.pure, ht, throw, throwThe, MonadExceptOf.throw] at h

/-- The policy's commitment: the record its bytes are, and its digest, SHA-512 of the record's format tag and its
canonical text. -/
theorem policy_of {b : ByteArray} {c : NciCheck.Protocol.Commitment} (h : NciCheck.Protocol.commitmentOf b = .ok c) :
    canonicalJson b = .ok c.record ∧ ∃ s, Flock.canon c.record = .ok s ∧
      c.digest = Flock.Sha512.hash ((if NciCheck.Baseline.strAt c.record "format" == some NciCheck.Baseline.FORMAT_V2
        then NciCheck.Baseline.FORMAT_V2 else NciCheck.Baseline.FORMAT).toUTF8.push 0 ++ s.toUTF8) := by
  unfold NciCheck.Protocol.commitmentOf at h
  rcases hj : canonicalJson b with e | j
  · simp [hj, Bind.bind, Except.bind] at h
  rcases hc : Flock.canon j with e | s
  · cases j <;> simp [hj, hc, Bind.bind, Except.bind, NciCheck.Baseline.digest, pure, Except.pure, throw, throwThe,
      MonadExceptOf.throw] at h
  cases j <;> simp [hj, hc, Bind.bind, Except.bind, NciCheck.Baseline.digest, pure, Except.pure, throw, throwThe,
      MonadExceptOf.throw, FlockComplete.JsonRT.unhex_hexOf] at h
  subst h
  exact ⟨rfl, s, hc, by simp⟩

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

theorem stage_getD (j : Json) (k : String) : SampledProofs.getD j k = NciCheck.Baseline.getD j k := rfl

theorem push_zero (x : ByteArray) : x.push 0 = x ++ ByteArray.mk #[0] := by
  apply ByteArray.ext
  simp [ByteArray.data_push, ByteArray.data_append]

/-- **One record, one digest**: bytes that the three subprotocols' commitment decoders all accept decode to one digest
under each of them. -/
theorem one_digest {b : ByteArray} {cI : NciCheck.Protocol.Commitment} {cW : PouwCheck.Registration}
    {cS : SampledProofs.OneStage.Commitment} (hI : NciCheck.Protocol.commitmentOf b = .ok cI)
    (hW : PouwCheck.registrationOf b = .ok cW) (hS : SampledProofs.OneStage.commitmentOf b = .ok cS) :
    cW.digest = cI.digest ∧ cS.digest = cI.digest := by
  obtain ⟨hjI, s, hs, hdI⟩ := policy_of hI
  obtain ⟨j, hjW, hf, hbytes⟩ := pouw_of hW
  obtain ⟨hjS, hdS⟩ := stage_of hS
  have e1 : cI.record = j := by rw [hjI] at hjW; exact Except.ok.inj hjW
  have e2 : cS.record = j := by rw [hjS] at hjW; exact Except.ok.inj hjW
  rw [e1] at hs hdI
  rw [e2] at hdS
  have hsb : s.toUTF8 = b := by
    obtain ⟨s', hs', hb'⟩ := canonical_text hjW
    rw [hs] at hs'
    cases hs'
    exact hb'
  simp only [Option.filter_eq_some_iff] at hf
  obtain ⟨hstr, hmem⟩ := hf
  have hget : NciCheck.Baseline.getD j "format" = .str cW.format := by
    rw [← pouw_get]
    revert hstr
    cases PouwCheck.get j "format" <;> simp [PouwCheck.str?]
  have hN : NciCheck.Baseline.strAt j "format" = some cW.format := by
    simp [NciCheck.Baseline.strAt, hget, NciCheck.Baseline.strOf?]
  have hSg : SampledProofs.getD j "format" = .str cW.format := by rw [stage_getD, hget]
  unfold SampledProofs.Registration.digest at hdS
  simp only [hSg, hs, Bind.bind, Except.bind, pure, Except.pure] at hdS
  have hdS' := (Except.ok.inj hdS).symm
  unfold PouwCheck.Registration.digest
  rw [hdI, hdS', hN, hbytes, ← hsb]
  have hfm : cW.format = NciCheck.Baseline.FORMAT ∨ cW.format = NciCheck.Baseline.FORMAT_V2 := by
    simpa [PouwCheck.formats, NciCheck.Baseline.FORMAT, NciCheck.Baseline.FORMAT_V2] using hmem
  have hv : SampledProofs.jeq (.str NciCheck.Baseline.FORMAT_V2) (.str NciCheck.Baseline.FORMAT_V2) = true := by
    simp [SampledProofs.jeq, canon_str, pure, Except.pure]
  have hv1 : SampledProofs.jeq (.str NciCheck.Baseline.FORMAT) (.str NciCheck.Baseline.FORMAT_V2) = false := by
    simp only [SampledProofs.jeq, canon_str, pure, Except.pure]
    decide +kernel
  have hS2 : SampledProofs.Registration.FORMAT_V2 = NciCheck.Baseline.FORMAT_V2 := rfl
  have hS1 : SampledProofs.Registration.FORMAT = NciCheck.Baseline.FORMAT := rfl
  rcases hfm with h1 | h2
  · have hne : (some NciCheck.Baseline.FORMAT == some NciCheck.Baseline.FORMAT_V2) = false := by decide +kernel
    rw [h1, hne, hS2, hS1]
    simp [hv1, push_zero]
  · rw [h2, hS2, hS1, hv]
    simp [push_zero]

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
