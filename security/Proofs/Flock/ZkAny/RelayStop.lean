module

public import Proofs.Flock.ZkAny.RelayFold
public import Proofs.Flock.Claims.ZK.FrameRun
public import Proofs.Flock.Claims.ZK.FrameStop

/-!
# A session's stop in the loop's frames

A session stops at its first failure, every rep with it: `Frames.Sessions.stopped` holds once some rep's frame of the
session stopped, and from then on `Frames.roundPlace` hands no rep's frame of it a device round. Here is what the loop's
frames' updates do to that bit:

* `stopped_iff`, `stopped_frameAt`, `stopped_of_frameAt`: the session stopped exactly where one of its reps' frames
  did; so where it has not, no frame the loop finds for it has, and a stopped frame it finds stops it.
* `stopped_put_self`, `stopped_put_live`, `stopped_put_other`: a put of a stopped frame at a rep's key stops the rep's
  session, one of a frame not stopped leaves a session not stopped so, and a put of another session's frame leaves the
  session's stop as it was.
* **`roundStep_stopped`**: so a round of the session that releases stops it exactly where it had stopped or the
  frame's check failed the round: a stopped session's rounds reach their frames as no device round, which the check
  fails (`Ranked.check_empty`), and a live one's frames are not stopped, so the check stops them only where it fails
  (`Ranked.check_stopped`).
* `SessKeptStep`, `SessKept`: every update but a put of one of the session's frames, a start of a stopped frame, and
  the session's own end, and any run of them; `stopped_sessKept`: those leave the session's stop as it was. The loop's
  updates between two of a session's requests are these: the other sessions' requests, and the frames port's, whose
  starts push a frame at the frame's start (`Frames.startRep`).
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (SessionSpec Stmt F128)
open _root_.Flock.Firewall.Frames (Sessions Rep)

/-- **The session stopped exactly where one of its reps' frames did.** -/
theorem stopped_iff {R : Sessions} {sid : Nat} :
    R.stopped sid = true ↔ ∃ p ∈ R.reps, p.sid = sid ∧ p.st.m.stopped = true := by
  unfold _root_.Flock.Firewall.Frames.Sessions.stopped
  rw [Array.any_eq_true']
  simp only [Bool.and_eq_true, beq_iff_eq]

/-- **Where the session has not stopped, no frame the loop finds for it has.** -/
theorem stopped_frameAt {R : Sessions} {sid i r : Nat} {p : Rep} (hs : R.stopped sid = false)
    (hf : frameAt R sid i r = some p) : p.st.m.stopped = false := by
  have hm := Array.mem_of_find?_eq_some hf
  have hk := Array.find?_some hf
  simp only [keyIs, Bool.and_eq_true, beq_iff_eq] at hk
  cases h : p.st.m.stopped
  · rfl
  · have : R.stopped sid = true := stopped_iff.mpr ⟨p, hm, hk.1.1, h⟩
    rw [hs] at this
    cases this

/-- **A stopped frame the loop finds for a session stops it.** -/
theorem stopped_of_frameAt {R : Sessions} {sid i r : Nat} {p : Rep} (hf : frameAt R sid i r = some p)
    (hp : p.st.m.stopped = true) : R.stopped sid = true := by
  have hk := Array.find?_some hf
  simp only [keyIs, Bool.and_eq_true, beq_iff_eq] at hk
  exact stopped_iff.mpr ⟨p, Array.mem_of_find?_eq_some hf, hk.1.1, hp⟩

/-- **A put of another session's frame** leaves the session's stop as it was. -/
theorem stopped_put_other {R : Sessions} {sid : Nat} {p : Rep} (hp : p.sid ≠ sid) :
    (R.put p).stopped sid = R.stopped sid := by
  unfold _root_.Flock.Firewall.Frames.Sessions.put _root_.Flock.Firewall.Frames.Sessions.stopped
  rw [Array.any_map]
  congr 1
  funext q
  simp only [Function.comp]
  split
  · rename_i h
    simp only [Bool.and_eq_true, beq_iff_eq] at h
    have : q.sid ≠ sid := h.1.1 ▸ hp
    simp only [beq_false_of_ne hp, beq_false_of_ne this, Bool.false_and]
  · rfl

/-- **A put of a stopped frame at a rep's key** stops the rep's session. -/
theorem stopped_put_self {R : Sessions} {p : Rep} (hfr : (frameAt R p.sid p.table p.rep).isSome)
    (hp : p.st.m.stopped = true) : (R.put p).stopped p.sid = true := by
  obtain ⟨q, hq⟩ := Option.isSome_iff_exists.mp hfr
  have hm := Array.mem_of_find?_eq_some hq
  have hk := Array.find?_some hq
  apply stopped_iff.mpr
  refine ⟨p, ?_, rfl, hp⟩
  unfold _root_.Flock.Firewall.Frames.Sessions.put
  simp only [Array.mem_map]
  refine ⟨q, hm, ?_⟩
  have hk' : (q.sid == p.sid && q.table == p.table && q.rep == p.rep) = true := hk
  simp only [hk', ↓reduceIte]

/-- **A put of a frame not stopped, in a session not stopped,** leaves the session not stopped. -/
theorem stopped_put_live {R : Sessions} {p : Rep} (hs : R.stopped p.sid = false) (hp : p.st.m.stopped = false) :
    (R.put p).stopped p.sid = false := by
  cases h : (R.put p).stopped p.sid
  · rfl
  · obtain ⟨q, hq, hqs, hqst⟩ := stopped_iff.mp h
    unfold _root_.Flock.Firewall.Frames.Sessions.put at hq
    simp only [Array.mem_map] at hq
    obtain ⟨q', hq', e⟩ := hq
    split at e
    · subst e
      rw [hp] at hqst
      cases hqst
    · subst e
      have : R.stopped p.sid = true := stopped_iff.mpr ⟨q', hq', hqs, hqst⟩
      rw [hs] at this
      cases this

/-- **An update that keeps session `sid`'s stop**: any of `RegStep`'s but a put of one of the session's frames, a
start of a stopped frame, and the session's own end. -/
inductive SessKeptStep (sid : Nat) : Sessions → Sessions → Prop
  | begin (R : Sessions) (sid' : Nat) (spec : SessionSpec) (sts : Array Stmt) :
      SessKeptStep sid R (R.begin sid' spec sts)
  | link (R : Sessions) (sid' : Nat) (c : Array F128) : SessKeptStep sid R (R.link sid' c)
  | start (R : Sessions) (q : Rep) : q.st.m.stopped = false → SessKeptStep sid R (startAt R q)
  | put (R : Sessions) (p : Rep) : p.sid ≠ sid → SessKeptStep sid R (R.put p)
  | keep (R : Sessions) (root : ByteArray) (r' : Nat) (s : Array ByteArray) : SessKeptStep sid R (keepOf R root r' s)
  | file {R R' : Sessions} (root : ByteArray) (r' : Nat) (b : ByteArray) : R.file root r' b = .ok R' →
      SessKeptStep sid R R'
  | inner {R R' : Sessions} (root : ByteArray) (r' j : Nat) (b : ByteArray) : R.innerRound root r' j b = .ok R' →
      SessKeptStep sid R R'
  | close (R : Sessions) (sid' : Nat) (roots : Array ByteArray) : sid' ≠ sid →
      SessKeptStep sid R (closeOf R sid' roots)

/-- Any run of updates that keep the session's stop. -/
inductive SessKept (sid : Nat) : Sessions → Sessions → Prop
  | refl (R : Sessions) : SessKept sid R R
  | step {R R' R'' : Sessions} : SessKept sid R R' → SessKeptStep sid R' R'' → SessKept sid R R''

/-- Each `SessKeptStep` is one of the loop's frames' updates. -/
theorem SessKeptStep.regStep {sid : Nat} {R R' : Sessions} (h : SessKeptStep sid R R') : RegStep R R' := by
  cases h with
  | begin => exact .begin _ _ _ _
  | link => exact .link _ _ _
  | start q _ => exact .start _ q
  | put p _ => exact .put _ p
  | keep => exact .keep _ _ _ _
  | file root r' b hf => exact .file root r' b hf
  | inner root r' j b hf => exact .inner root r' j b hf
  | close sid' roots _ => exact .close _ sid' roots

/-- **An update that keeps the session's stop keeps it.** -/
theorem stopped_sessKeptStep {sid : Nat} {R R' : Sessions} (h : SessKeptStep sid R R') :
    R'.stopped sid = R.stopped sid := by
  cases h with
  | begin => rfl
  | link => rfl
  | start q hq =>
    unfold startAt
    split
    · rfl
    · unfold _root_.Flock.Firewall.Frames.Sessions.stopped
      simp [hq]
  | put p hp => exact stopped_put_other hp
  | keep =>
    unfold keepOf
    split <;> rfl
  | file root r' b hf =>
    unfold _root_.Flock.Firewall.Frames.Sessions.file at hf
    split at hf
    · cases hf
    · cases hf
      rfl
  | inner root r' j b hf =>
    unfold _root_.Flock.Firewall.Frames.Sessions.innerRound at hf
    simp only at hf
    split at hf
    · cases hf
    · split at hf <;> (cases hf; rfl)
  | close sid' roots hne =>
    apply Bool.eq_iff_iff.mpr
    rw [stopped_iff, stopped_iff]
    constructor
    · rintro ⟨p, hm, hs, hp⟩
      exact ⟨p, (Array.mem_filter.mp hm).1, hs, hp⟩
    · rintro ⟨p, hm, hs, hp⟩
      refine ⟨p, Array.mem_filter.mpr ⟨hm, ?_⟩, hs, hp⟩
      simp [hs, Ne.symm hne]

/-- **A run of updates that keep the session's stop keeps it.** -/
theorem stopped_sessKept {sid : Nat} {R R' : Sessions} (h : SessKept sid R R') : R'.stopped sid = R.stopped sid := by
  induction h with
  | refl => rfl
  | step _ hs ih => exact (stopped_sessKeptStep hs).trans ih

/-- **A released round of the session stops it exactly where it had stopped or the frame's check failed the round**:
at a round at `roundPlace`'s scheduled place `rr` on the frame `q`, where a stopped session takes no device round and a
live one's frame is not stopped, the frame the round puts back (`p₁`) is the frame's check's, and after its put the
session is stopped where it was or where the check failed the round. -/
theorem roundStep_stopped {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r : Nat}
    {dom content : ByteArray} {n : Nat} {q p₁ : Rep} {rr : ByteArray} {f : Bool} {nd : Option String}
    {o : ByteArray} (hfr : (frameAt R sid i r).isSome)
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, .sched rr, f, nd))
    (hnd : R.stopped sid = true → nd.isSome = true) (hq : R.stopped sid = false → q.st.m.stopped = false)
    (ho : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).out = .ok o)
    (hput : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).put = some p₁) :
    ∃ out why, _root_.Flock.Firewall.Frame.check q.cfg q.st.m (if nd.isSome then ByteArray.empty else content) rr n =
        .ok (out, p₁.st.m, why) ∧ (R.put p₁).stopped sid = (R.stopped sid || why.isSome) := by
  rcases hres : _root_.Flock.Firewall.Frame.step q.cfg q.st
    (.round n (if nd.isSome then ByteArray.empty else content) (.sched rr)) with ⟨a, st'⟩
  obtain ⟨v, hv⟩ := roundStep_put hrp hres
  rw [hv, Option.some.injEq] at hput
  have hst₁ : p₁.st = st' := by
    subst hput
    rfl
  have hk : keyOf p₁ = (sid, i, r) := by
    rw [← roundPlace_key hrp]
    subst hput
    rfl
  simp only [keyOf, Prod.mk.injEq] at hk
  obtain ⟨hs, ht, hr⟩ := hk
  obtain ⟨out, vd, hre, -⟩ := roundStep_out hrp ho
  rw [hres] at hre
  dsimp only at hre
  subst hre
  obtain ⟨why, hc⟩ := _root_.Flock.Firewall.Frame.check_of_step hres
  rw [← hst₁] at hc
  refine ⟨out, why, hc, ?_⟩
  have hfr' : (frameAt R p₁.sid p₁.table p₁.rep).isSome := by
    rw [hs, ht, hr]
    exact hfr
  have hst := Ranked.check_stopped hc
  cases hR : R.stopped sid
  · rw [hq hR, Bool.false_or] at hst
    cases hw : why.isSome
    · rw [hw] at hst
      have e := stopped_put_live (R := R) (p := p₁) (by rw [hs]; exact hR) hst
      rw [hs] at e
      simpa using e
    · rw [hw] at hst
      have e := stopped_put_self hfr' hst
      rw [hs] at e
      simpa using e
  · rw [ite_eq_left (hnd hR)] at hc
    have e := stopped_put_self hfr' (Ranked.check_empty hc)
    rw [hs] at e
    simpa using e

end Claims.ZKServer.Relay
