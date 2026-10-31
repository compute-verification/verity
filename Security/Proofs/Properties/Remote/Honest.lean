import Proofs.Properties.Remote
import Proofs.Flock.Claims.Complete.Json.Hex

/-!
# Remote's tie on an honest run: one record, one digest

The honest lemma of `AcceptInferenceOnly`'s tie: a run commits the policy and the one-stage audit to its one
registration record, and whenever both commitment decoders accept that record's bytes, they decode them to one digest
(`one_digest`), so no record strays and both digests are the policy's record's (`acceptInferenceOnly_honest`).
The record may name any registered circuit, a tiny one among them: the commitment decoders read only its `format`.

The lemma takes the decoders' acceptance of the bytes as its hypotheses rather than exhibiting concrete bytes: the
decoders read their bytes through `Lean.Json.parse`, which is `partial`, so no proof evaluates them on a literal.
-/

namespace Properties.Remote.SecurityProofs

open Core Core.Composition
open Lean (Json)

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

/-- **One record, one digest**: bytes that the policy's and the one-stage audit's commitment decoders both accept decode
to one digest under each. -/
theorem one_digest {b : ByteArray} {cI : NciCheck.Protocol.Commitment} {cS : SampledProofs.OneStage.Commitment}
    (hI : NciCheck.Protocol.commitmentOf b = .ok cI) (hS : SampledProofs.OneStage.commitmentOf b = .ok cS) :
    cS.digest = cI.digest := by
  obtain ⟨hjI, hdI⟩ := policy_of hI
  obtain ⟨hjS, hdS⟩ := stage_of hS
  have e : cS.record = cI.record := by rw [hjI] at hjS; exact (Except.ok.inj hjS).symm
  rw [e] at hdS
  exact Except.ok.inj (hdS.symm.trans hdI)

/-- **Remote's tie holds on an honest run** (the honest lemma of `AcceptInferenceOnly`'s tie): the policy's and the
one-stage audit's cases committing to one record's bytes, which both commitment decoders accept, leave no stray among
the components remote's check compares (`remote_check_ids`), and both digests are the policy's record's. -/
theorem acceptInferenceOnly_honest {kI kW kS : Case} {cI : NciCheck.Protocol.Commitment}
    {cS : SampledProofs.OneStage.Commitment} (hS' : kS.commitment = kI.commitment)
    (hI : NciCheck.Protocol.commitmentOf kI.commitment = .ok cI)
    (hS : SampledProofs.OneStage.commitmentOf kI.commitment = .ok cS) :
    strays [NciCheck.Protocol.protocol.id, SampledProofs.OneStage.protocol.id]
        [NciCheck.Protocol.protocol, PouwCheck.Period.protocol, SampledProofs.OneStage.protocol] [kI, kW, kS] = none ∧
      Properties.Composition.digestOf NciCheck.Protocol.protocol kI.commitment =
        some (NciCheck.Protocol.protocol.digest cI) ∧
      Properties.Composition.digestOf SampledProofs.OneStage.protocol kS.commitment =
        some (NciCheck.Protocol.protocol.digest cI) := by
  have dS := one_digest hI hS
  have hcI : NciCheck.Protocol.protocol.commitment kI.commitment = .ok cI := hI
  have hcS : SampledProofs.OneStage.protocol.commitment kS.commitment = .ok cS := by rw [hS']; exact hS
  have dS' : SampledProofs.OneStage.protocol.digest cS = NciCheck.Protocol.protocol.digest cI := dS
  exact ⟨strays_two_none ne_policy_stage ne_period_stage hcI hcS dS', digestOf_ok hcI,
    (digestOf_ok hcS).trans (congrArg some dS')⟩

end Properties.Remote.SecurityProofs
