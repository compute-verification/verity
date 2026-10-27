import Proofs.Properties.Verity.SampledCanon
import Proofs.Flock.Soundness.Discharge.ZkExec.Verify

/-!
# The coupling's transcript and the live outcome it is a record of

`Drive.proj` reads the play's outcome as session 0's transcript of v2's evidence (`Canon.proj`, `Canon.transcriptOf`):
where onsite's sampled case decodes to `(p, c, e)`, v2's `checkBlobs` accepts it and `e`'s first session is `s0`, the
record the blobs hold under `s0`'s digest and the proof files `verify --zk` reads for it (`SessionTranscript`). It
reads only what the prover sent (`Outcome.sent`), so the reduction computes it at the prover's last message, where
`Drive.pfOf` gives the live audit the roots the record decodes to at its setup (`Canon.rootsOf`) and the proof files
(`Canon.pfOf`).

The live outcome's roots are then the record's own, so the record matches them by construction
(`Canon.recordOfZKJ_proj`): the record is the live outcome's (`Flock.Guarantees.RecordOfZKJ`) wherever its draw is the
unit draw the releases read back to (`Canon.DrawAt`), and its link points and every rep stream's rounds are what the
releases read back to (`Canon.PointsAt`, `Canon.RoundsAt`). The roots meet the play only through SHA-512: where the
link points' request carries what session 0's first use of the challenger's coins is released after
(`Canon.PointsAfter`), that is `LiveHash.pointsHash` of the roots (`Canon.pointsHash_pfOf`), so no raw root is compared
with anything the play holds.
-/

open Lean (Json)

namespace Verity.Canon

open FlockSoundness.Discharge
open Properties.SampledProofs.OneStageZk (SessionInputs SessionTranscript acceptsZK_of_transcript)

local notation "𝔽" => FlockLevel3.GF128

/-! ## The transcript the coupling reads -/

/-- **Session 0's transcript of what was sent**: onsite's sampled case decodes to v2's `(p, c, e)`, `checkBlobs`
accepts it, and `t` is the transcript of `e`'s first session. -/
def Sess0 (s : Sent) (t : Exec.Transcript) : Prop :=
  ∃ p c e pub prof s0 rest, SampledCase s p c e ∧
    _root_.SampledProofs.OneStageZk.checkBlobs p c e (Core.Blobs.ofArray s.blobs) = .accept pub prof ∧
    e.sessions.toList = s0 :: rest ∧ SessionTranscript p e s.blobs s0 t

open Classical in
/-- **The coupling's transcript of what was sent**: session 0's (`Sess0`), where there is one. -/
noncomputable def transcriptOf (s : Sent) : Exec.Transcript :=
  if h : ∃ t, Sess0 s t then Classical.choose h else ⟨"", #[]⟩

theorem transcriptOf_spec {s : Sent} (h : ∃ t, Sess0 s t) : Sess0 s (transcriptOf s) := by
  unfold transcriptOf
  rw [dif_pos h]
  exact Classical.choose_spec h

/-- **Where v2 accepts, session 0 has a transcript.** -/
theorem sess0_of_accept {s : Sent} {p : _root_.SampledProofs.OneStageZk.Params}
    {c : _root_.SampledProofs.OneStage.Commitment} {e : _root_.SampledProofs.OneStageZk.Evidence}
    {pub : _root_.SampledProofs.OneStage.Public} {prof : Core.Integrity} (hc : SampledCase s p c e)
    (hacc : _root_.SampledProofs.OneStageZk.checkBlobs p c e (Core.Blobs.ofArray s.blobs) = .accept pub prof)
    {s0 : _root_.SampledProofs.OneStageZk.Session} {rest : List _root_.SampledProofs.OneStageZk.Session}
    (hl : e.sessions.toList = s0 :: rest) : ∃ t, Sess0 s t := by
  have hs : s0 ∈ e.sessions := Array.mem_toList_iff.1 (by rw [hl]; exact List.mem_cons_self)
  obtain ⟨tags0, circuitB, pubB, rd, b, record, proofs, ht, hcB, hpB, hrd, hb, hu, hf, -⟩ :=
    Properties.SampledProofs.OneStageZk.acceptsZK_of_session hacc hs
  exact ⟨⟨record, proofs⟩, p, c, e, pub, prof, s0, rest, hc, hacc, hl,
    tags0, circuitB, pubB, rd, b, ht, hcB, hpB, hrd, hb, hu, hf⟩

/-- **Session 0's transcript is accepted** at inputs that are the session's wherever v2 accepts (`Drive.covers`'
`SessionInputs`). -/
theorem acceptsZK_of_sess0 {s : Sent} {t : Exec.Transcript} {I : Exec.Inputs} (h : Sess0 s t)
    (hI : ∀ p c e pub prof, SampledCase s p c e →
      _root_.SampledProofs.OneStageZk.checkBlobs p c e (Core.Blobs.ofArray s.blobs) = .accept pub prof →
      SessionInputs p e s.blobs I) :
    ZkExec.AcceptsZK I t := by
  obtain ⟨p, c, e, pub, prof, s0, rest, hc, hacc, hl, ht⟩ := h
  have hs : s0 ∈ e.sessions := Array.mem_toList_iff.1 (by rw [hl]; exact List.mem_cons_self)
  exact acceptsZK_of_transcript hacc hs ht (hI p c e pub prof hc hacc)

/-- What the prover sent, at the parameters `θ`, the committed file `F`, the commitment `c` and its last message `e`
(`Outcome.sent`). -/
def sentAt (θ F c : ByteArray) (e : ByteArray × Array ByteArray) : Sent := ⟨θ, c, e.1, #[F] ++ e.2⟩

theorem sent_eq (θ F c : ByteArray) (log : List Release) (e : ByteArray × Array ByteArray) :
    (⟨θ, F, c, log, e.1, e.2⟩ : Outcome).sent = sentAt θ F c e := rfl

/-- **`Drive.proj`**: the coupling's transcript of the play's outcome, read off what the prover sent. -/
noncomputable def proj (o : Outcome) : Exec.Transcript := transcriptOf o.sent

/-! ## The roots and proof files the reduction sends last -/

/-- **The roots a record decodes to** at the `--zk` setup of `I` at its draw (`Record.decode`'s `rootB`); none where it
sets up or decodes to nothing. -/
noncomputable def rootsOf (I : Exec.Inputs) (record : String) : Array ByteArray :=
  match Exec.setupOf (ZkExec.zkInputs I) record with
  | .ok (spec, _) =>
    match Flock.Record.decode spec record with
    | .ok sess => sess.rootB
    | .error _ => #[]
  | .error _ => #[]

theorem rootsOf_eq {I : Exec.Inputs} {record : String} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    {sess : Flock.Session} (hs : Exec.setupOf (ZkExec.zkInputs I) record = .ok (spec, sts))
    (hd : Flock.Record.decode spec record = .ok sess) : rootsOf I record = sess.rootB := by
  first
    | (simp only [rootsOf, hs, hd]; done)
    | (unfold rootsOf; rw [hs]; dsimp only; rw [hd])
    | (unfold rootsOf; rw [hs]; simp only [hd])

/-- **`Drive.pfOf`**: at the prover's last message `e`, the roots session 0's record decodes to at `I`'s setup, and its
proof files. -/
noncomputable def pfOf (I : Exec.Inputs) (θ F c : ByteArray) (e : ByteArray × Array ByteArray) :
    Flock.Guarantees.LiveHash.Last :=
  (rootsOf I (transcriptOf (sentAt θ F c e)).record, (transcriptOf (sentAt θ F c e)).proofs)

theorem pfOf_eq (I : Exec.Inputs) (θ F c : ByteArray) (log : List Release) (e : ByteArray × Array ByteArray) :
    pfOf I θ F c e = (rootsOf I (proj ⟨θ, F, c, log, e.1, e.2⟩).record, (proj ⟨θ, F, c, log, e.1, e.2⟩).proofs) :=
  rfl

/-! ## What the releases establish of the record -/

/-- **The record's draw is the unit draw at `S`**: what the draw's releases read back to. -/
def DrawAt {n : ℕ} (dj : Finset (Fin n) → Json) (S : Finset (Fin n)) (t : Exec.Transcript) : Prop :=
  Exec.recordDraw t.record = some (dj S)

/-- **The record's link points are `pts`**, at every `--zk` setup of `I` and decode of the record: what the link
points' releases read back to. -/
def PointsAt {mPts : ℕ} (I : Exec.Inputs) (t : Exec.Transcript) (pts : Fin 2 → Fin mPts → 𝔽) : Prop :=
  ∀ spec sts sess, Exec.setupOf (ZkExec.zkInputs I) t.record = .ok (spec, sts) →
    Flock.Record.decode spec t.record = .ok sess → FlockSoundness.Refine.ptsOf mPts sess.points = pts

/-- **The record's rep streams' rounds are `rs`**, at every `--zk` setup of `I` and decode of the record: table `j`'s
rep `r` is a stream of the record whose rounds are `rs j r`, what the rounds' releases read back to. -/
def RoundsAt {J : ℕ} (I : Exec.Inputs) (t : Exec.Transcript) (rs : Fin J → Fin 2 → List Flock.Round) : Prop :=
  ∀ spec sts sess, Exec.setupOf (ZkExec.zkInputs I) t.record = .ok (spec, sts) →
    Flock.Record.decode spec t.record = .ok sess → ∀ (j : Fin J) (r : Fin 2), ∃ s : Flock.StreamRec,
      sess.streams.find? (fun x => x.name == toString spec.tables[j.val]! ++ toString "/rep" ++ toString r.val) =
        some s ∧ s.rounds.toList = rs j r

/-- **The link points' request carries `h`**: session 0's first use of the challenger's coins (`sessionUses 0`, its
link points) is released after `h`. -/
def PointsAfter (t : Exec.Transcript) (h : ByteArray) : Prop :=
  ∃ us u w, _root_.SampledProofs.OneStageZk.sessionUses 0 t.record = .ok us ∧ us.head? = some (u, w) ∧ u.after = h

/-! ## The record of the live outcome -/

/-- **A transcript is a record of the live outcome at its own roots**: at the roots it decodes to (`rootsOf`) and its
proof files, the transcript is a record of the live outcome (`RecordOfZKJ`) wherever its draw, link points and rounds
are the outcome's. -/
theorem recordOfZKJ_rootsOf {n mPts J : ℕ} {I : Exec.Inputs} {dj : Finset (Fin n) → Json} {S : Finset (Fin n)}
    {t : Exec.Transcript} {pts : Fin 2 → Fin mPts → 𝔽} {rs : Fin J → Fin 2 → List Flock.Round}
    (hDraw : DrawAt dj S t) (hPts : PointsAt I t pts) (hRounds : RoundsAt I t rs) :
    Flock.Guarantees.RecordOfZKJ I dj J S ⟨rootsOf I t.record, pts, rs, t.proofs⟩ t := by
  unfold Flock.Guarantees.RecordOfZKJ
  refine ⟨hDraw, rfl, fun spec sts sess hs hd => ⟨hPts spec sts sess hs hd, fun j => ⟨?_, hRounds spec sts sess hs hd j⟩⟩⟩
  show sess.rootB[j.val]! = (rootsOf I t.record)[j.val]!
  rw [rootsOf_eq hs hd]

/-- **`Drive.canon`'s record, at `proj` and `pfOf`**: the coupling's transcript of the play is a record of the live
outcome at the roots and proof files the reduction sends last, its link points `pts` and rounds `rs`, wherever the
transcript's draw, link points and rounds are those the releases read back to. -/
theorem recordOfZKJ_proj {n mPts J : ℕ} {I : Exec.Inputs} {dj : Finset (Fin n) → Json} {S : Finset (Fin n)}
    {θ F c : ByteArray} {log : List Release} {e : ByteArray × Array ByteArray}
    {pts : Fin 2 → Fin mPts → 𝔽} {rs : Fin J → Fin 2 → List Flock.Round}
    (hDraw : DrawAt dj S (proj ⟨θ, F, c, log, e.1, e.2⟩))
    (hPts : PointsAt I (proj ⟨θ, F, c, log, e.1, e.2⟩) pts)
    (hRounds : RoundsAt I (proj ⟨θ, F, c, log, e.1, e.2⟩) rs) :
    Flock.Guarantees.RecordOfZKJ I dj J S ⟨(pfOf I θ F c e).1, pts, rs, (pfOf I θ F c e).2⟩
      (proj ⟨θ, F, c, log, e.1, e.2⟩) :=
  recordOfZKJ_rootsOf hDraw hPts hRounds

/-! ## The roots, through their hash -/

/-- **The link points' request carries `pointsHash` of the roots the record decodes to**: at a session's inputs, where
flock-verify `--zk` accepts the transcript and the link points are released after `h`. -/
theorem pointsHash_rootsOf {p : _root_.SampledProofs.OneStageZk.Params} {e : _root_.SampledProofs.OneStageZk.Evidence}
    {blobs : Array ByteArray} {I : Exec.Inputs} {t : Exec.Transcript} (hI : SessionInputs p e blobs I)
    (hacc : ZkExec.AcceptsZK I t) {h : ByteArray} (hreq : PointsAfter t h) :
    Flock.Guarantees.LiveHash.pointsHash (rootsOf I t.record) = h := by
  obtain ⟨spec, sts, hs, -, hv⟩ := hacc
  obtain ⟨-, sess, -, hd, -⟩ := ZkExec.zk_verify_ok _ _ _ _ hv
  obtain ⟨tags0, circuitB, pubB, ht, -, -, rfl⟩ := hI
  obtain ⟨htab, hsha⟩ := _root_.SampledProofs.OneStageZk.sessionI_spec ht hs
  obtain ⟨us, u, w, hus, hhd, rfl⟩ := hreq
  obtain ⟨pts, hhd'⟩ := _root_.SampledProofs.OneStageZk.sessionUses_pointsHash_of_decode hus hd htab hsha
  rw [hhd] at hhd'
  obtain ⟨rfl, -⟩ := Prod.mk.inj (Option.some.inj hhd')
  first
    | (rw [rootsOf_eq hs hd]; done)
    | (rw [rootsOf_eq hs hd]; rfl)
    | simp only [rootsOf_eq hs hd]

/-- **`Drive.canon`'s hash, at `proj` and `pfOf`**: where session 0's transcript of the play is accepted at inputs that
are the session's wherever v2 accepts, and the link points' request carries `h`, the roots the reduction sends last
hash to `h`. -/
theorem pointsHash_pfOf {I : Exec.Inputs} {θ F c : ByteArray} {log : List Release} {e : ByteArray × Array ByteArray}
    (hS : ∃ t, Sess0 (sentAt θ F c e) t)
    (hI : ∀ p c' e' pub prof, SampledCase (sentAt θ F c e) p c' e' →
      _root_.SampledProofs.OneStageZk.checkBlobs p c' e' (Core.Blobs.ofArray (sentAt θ F c e).blobs) =
        .accept pub prof → SessionInputs p e' (sentAt θ F c e).blobs I)
    {h : ByteArray} (hreq : PointsAfter (proj ⟨θ, F, c, log, e.1, e.2⟩) h) :
    Flock.Guarantees.LiveHash.pointsHash (pfOf I θ F c e).1 = h := by
  obtain ⟨p, c', e', pub, prof, s0, rest, hc, hacc, hl, ht⟩ := transcriptOf_spec hS
  have hs : s0 ∈ e'.sessions := Array.mem_toList_iff.1 (by rw [hl]; exact List.mem_cons_self)
  exact pointsHash_rootsOf (hI p c' e' pub prof hc hacc)
    (acceptsZK_of_transcript hacc hs ht (hI p c' e' pub prof hc hacc)) hreq

end Verity.Canon
