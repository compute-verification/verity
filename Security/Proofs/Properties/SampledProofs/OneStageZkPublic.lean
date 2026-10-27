import SampledProofs.OneStageZk

/-!
# v2 accepts only the public file its registration names

`OneStageZk.checkBlobs` refuses evidence whose public file is not the one the registration names (`publicWhy`) before
it runs `checkProofs`. So where it accepts, the evidence's public digest is the registration's `window.public_file`
(`public_of_accept`), and `checkProofs` accepts the same outcome (`checkProofs_of_accept`): every file a session reads
as its public file is fetched under the digest the statement registers, never one the evidence alone chose.
-/

namespace SampledProofs.OneStageZk

/-- `publicWhy` names no reason only where the registration names the public file `pub`. -/
theorem registeredPublic_of_publicWhy {c : OneStage.Commitment} {pub : String} (h : publicWhy c pub = none) :
    registeredPublic c = some pub := by
  unfold publicWhy at h
  cases hd : registeredPublic c with
  | none =>
    rw [hd] at h
    cases h
  | some d =>
    rw [hd] at h
    by_cases hp : pub = d
    · rw [hp]
    · simp [hp] at h

/-- **v2 accepts only the public file the registration names**: where `checkBlobs` accepts, the evidence's public
digest is the registration's `window.public_file`. -/
theorem public_of_accept {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    {x : OneStage.Public} {y : Core.Integrity} (h : checkBlobs p c e bs = .accept x y) :
    registeredPublic c = some e.pub := by
  unfold checkBlobs at h
  cases hw : publicWhy c e.pub with
  | some w =>
    rw [hw] at h
    cases h
  | none => exact registeredPublic_of_publicWhy hw

/-- Where `checkBlobs` accepts, `checkProofs` accepts the same outcome. -/
theorem checkProofs_of_accept {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    {x : OneStage.Public} {y : Core.Integrity} (h : checkBlobs p c e bs = .accept x y) :
    checkProofs p c e bs = .accept x y := by
  unfold checkBlobs at h
  cases hw : publicWhy c e.pub with
  | some w =>
    rw [hw] at h
    cases h
  | none =>
    rw [hw] at h
    exact h

end SampledProofs.OneStageZk
