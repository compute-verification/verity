import Proofs.Flock.Claims.ZK.FrameRanked
import Proofs.Flock.Claims.ZK.FrameCount

/-!
# The algebraic phase's rounds pass over the frame's phase

The frame moves its phase only at four labels (`Masker.label`): the opening's (`LABEL_PCS`), level 0's close and open,
and the inner proof's. A rep's algebraic phase (`Skeleton.rounds`, the masked zerocheck and lincheck) frames none of
them, so past its first round each of its scheduled rounds is plain frames and a squeeze:

* `plainLabel`, `Plain`: a label other than those four, and a frame a round holds before its squeeze (`midFrame`) that
  is no other label.
* `QPend`, `QRound`: pending bytes of plain frames, and a round of plain frames and a squeeze of its count, as
  `roundFree_encs` takes it.
* `quietS`: the simulation that keeps `QPend` of a recording's transcript and records only `QRound`s, with its
  transcript steps (`label_quiet`, `absorbF_quiet`, `absorbSlice_quiet`, `absorbBytes_quiet`, `squeezeP_quiet`).
* **`rounds_quiet`**: every scheduled round of the algebraic phase but its first (the zerocheck's head, which
  `Ranked.run_head` reads) is a `QRound`.
-/

namespace Claims.ZKServer.Quiet

open _root_.Flock (V F128 Transcript TMW liftTM)
open _root_.Flock.Transcript (header framed label absorbF absorbSlice absorbBytes)
open _root_.Flock.Firewall.Skeleton (RecM recAsks)
open _root_.Flock.Firewall.Finish (squeezeP sampleP sampleVecP zerocheckFw lincheckFw repPiop)
open _root_.Flock.Firewall.Frame (isLabel LABEL_PCS)
open FlockSoundness.Refine (Frame encs)

/-- **A label the frame passes over**: none of the four that move its phase (`Masker.label`), the opening's, level 0's
close, the inner proof's and level 0's open. -/
def plainLabel (b : ByteArray) : Bool :=
  !isLabel b LABEL_PCS && !isLabel b _root_.Flock.Zk.LABEL_CLOSE && !isLabel b _root_.Flock.Zk.LABEL_INNER &&
    !isLabel b _root_.Flock.Zk.LABEL_OPEN

/-- A frame a round holds before its squeeze (`midFrame`), and no label that moves the frame's phase. -/
def Plain (f : Frame) : Prop := midFrame f ∧ ∀ b, f = .label b → plainLabel b = true

/-- Pending bytes of plain frames. -/
def QPend (p : ByteArray) : Prop := ∃ fs, p.data.toList = encs fs ∧ ∀ f ∈ fs, Plain f

/-- **A quiet round**: plain frames, then a squeeze of the round's count `r.2 ≠ 0`, as `roundFree_encs` takes it. -/
def QRound (r : ByteArray × ℕ) : Prop :=
  ∃ fs k, r.1.data.toList = encs fs ++ (Frame.sq k r.2).enc ∧ (∀ f ∈ fs, Plain f) ∧ (k = 2 ∨ (k = 1 ∧ r.2 = 1)) ∧
    r.2 ≠ 0 ∧ r.2 < 2 ^ 64

theorem qpend_snoc {p q : ByteArray} {f : Frame} (h : QPend p) (hf : Plain f)
    (hq : q.data.toList = p.data.toList ++ f.enc) : QPend q := by
  obtain ⟨fs, hfs, hall⟩ := h
  refine ⟨fs ++ [f], by rw [hq, hfs, Sched.encs_snoc], fun g hg => ?_⟩
  rcases List.mem_append.1 hg with hg | hg
  · exact hall g hg
  · rw [List.mem_singleton.1 hg]; exact hf

theorem qpend_empty : QPend ByteArray.empty := ⟨[], rfl, fun _ h => nomatch h⟩

/-- **A recording program keeps quiet**: from plain pending bytes, wherever it returns, its pending bytes are plain and
the rounds it recorded are quiet. -/
def Quiet {α : Type} (p : TMW RecM α) : Prop :=
  ∀ (t : Transcript) (s : Array (ByteArray × ℕ) × ℕ) (a : α) (t' : Transcript) (s' : Array (ByteArray × ℕ) × ℕ),
    (p.run t).run s = .ok ((a, t'), s') → QPend t.pending →
      QPend t'.pending ∧ ∃ new : List (ByteArray × ℕ), s'.1.toList = s.1.toList ++ new ∧ ∀ r ∈ new, QRound r

/-- **The simulation of `Quiet`**: a recording program is related to itself when it keeps quiet. -/
def quietS : _root_.Flock.With.Sim (TMW RecM) (TMW RecM) where
  R _ y := Quiet y
  pure a t s a' t' s' h hp := by
    rw [StateT.run_pure, StateT.run_pure] at h
    cases h
    exact ⟨hp, [], by simp, fun _ h => nomatch h⟩
  bind {α β x y f g} hxy hfg t s b t'' s'' h hp := by
    rw [StateT.run_bind] at h
    obtain ⟨⟨a, t₁⟩, s₁, h1, h2⟩ := Count.rec_bind_ok h
    obtain ⟨hp1, n1, e1, q1⟩ := hxy t s a t₁ s₁ h1 hp
    obtain ⟨hp2, n2, e2, q2⟩ := hfg a t₁ s₁ b t'' s'' h2 hp1
    refine ⟨hp2, n1 ++ n2, by rw [e2, e1, List.append_assoc], fun r hr => ?_⟩
    rcases List.mem_append.1 hr with hr | hr
    · exact q1 r hr
    · exact q2 r hr
  throw e t s a t' s' h := by cases h
  lift v t s a t' s' h hp := by
    cases v with
    | error e => cases h
    | ok b =>
      cases h
      exact ⟨hp, [], by simp, fun _ h => nomatch h⟩

/-! ## The transcript's steps -/

/-- A verifier action that appends one plain frame keeps quiet. -/
theorem liftTM_quiet {α : Type} (x : _root_.Flock.Transcript.TM α)
    (hx : ∀ t a t', x t = .ok (a, t') → ∃ f, Plain f ∧ t'.pending.data.toList = t.pending.data.toList ++ f.enc) :
    quietS.R (liftTM x : TMW RecM α) (liftTM x) := by
  intro t s a t' s' h hp
  have e : ((liftTM x : TMW RecM α).run t).run s = (do let r ← x t; pure (r, s)) := rfl
  rw [e] at h
  rcases hx' : x t with _ | ⟨a', t''⟩
  · rw [hx'] at h; cases h
  · rw [hx'] at h
    cases h
    obtain ⟨f, hf, hq⟩ := hx t _ _ hx'
    exact ⟨qpend_snoc hp hf hq, [], by simp, fun _ h => nomatch h⟩

theorem label_quiet (l : String) (hl : l.toUTF8.size < 2 ^ 64) (hp : plainLabel l.toUTF8 = true) :
    quietS.R (liftTM (label l) : TMW RecM Unit) (liftTM (label l)) :=
  liftTM_quiet _ fun t _ _ h => by
    cases h
    refine ⟨.label l.toUTF8, ⟨hl, fun b e => ?_⟩, FlockSoundness.Refine.label_enc _ _⟩
    rw [← Frame.label.inj e]
    exact hp

theorem absorbF_quiet (v : F128) : quietS.R (liftTM (absorbF v) : TMW RecM Unit) (liftTM (absorbF v)) :=
  liftTM_quiet _ fun t _ _ h => by
    cases h
    exact ⟨.one v, ⟨trivial, fun b e => by cases e⟩, FlockSoundness.Refine.absorbF_enc _ _⟩

theorem absorbSlice_quiet (vs : Array F128) (hn : vs.size < 2 ^ 64) :
    quietS.R (liftTM (absorbSlice vs) : TMW RecM Unit) (liftTM (absorbSlice vs)) :=
  liftTM_quiet _ fun t _ _ h => by
    cases h
    exact ⟨.vec vs.toList, ⟨show vs.toList.length < 2 ^ 64 by simpa using hn, fun b e => by cases e⟩,
      FlockSoundness.Refine.absorbSlice_enc _ _⟩

theorem absorbSlice_quiet64 (f : ℕ → F128) :
    quietS.R (liftTM (absorbSlice ((Array.range 64).map f)) : TMW RecM Unit)
      (liftTM (absorbSlice ((Array.range 64).map f))) :=
  absorbSlice_quiet _ (by simp)

theorem absorbBytes_quiet (b : ByteArray) (hn : b.size < 2 ^ 64) :
    quietS.R (liftTM (absorbBytes b) : TMW RecM Unit) (liftTM (absorbBytes b)) :=
  liftTM_quiet _ fun t _ _ h => by
    cases h
    exact ⟨.bytes b, ⟨hn, fun b' e => by cases e⟩, FlockSoundness.Refine.absorbBytes_enc _ _⟩

/-- **A squeeze keeps quiet**: of `0` coins its frame stays pending; else it records a quiet round of `n`. -/
theorem squeezeP_quiet (kind : UInt8) (n : ℕ) (hk : kind = 2 ∨ (kind = 1 ∧ n = 1)) (hn : n < 2 ^ 64) :
    quietS.R (squeezeP recAsks kind n) (squeezeP recAsks kind n) := by
  intro t s a t' s' h hp
  obtain ⟨rs, k⟩ := s
  by_cases h0 : n = 0
  · subst h0
    have e : ((squeezeP recAsks kind 0).run t).run (rs, k) =
        .ok ((#[], { t with pending := Transcript.framed t kind 0 }), (rs, k)) := rfl
    rw [e] at h
    cases h
    have hk2 : kind = 2 := hk.resolve_right fun h => absurd h.2 (by decide)
    exact ⟨qpend_snoc (f := .sq kind 0) hp ⟨show kind = 2 ∧ (0 : ℕ) = 0 from ⟨hk2, rfl⟩, fun b e => by cases e⟩
      (FlockSoundness.Refine.framed_enc t kind 0), [],
      by simp, fun _ h => nomatch h⟩
  · have e : ((squeezeP recAsks kind n).run t).run (rs, k) =
        .ok (((Array.range n).map fun i => _root_.Flock.Firewall.Skeleton.coin (k + i + 1),
          { t with pending := ByteArray.empty, k := t.k + 1,
                   rounds := t.rounds.push ⟨n, ByteArray.empty,
                     (Array.range n).map fun i => _root_.Flock.Firewall.Skeleton.coin (k + i + 1),
                     some (Transcript.framed t kind n)⟩ }),
          (rs.push (Transcript.framed t kind n,
            (_root_.Flock.u64OfLE (Transcript.framed t kind n) ((Transcript.framed t kind n).size - 8)).toNat),
           k + (_root_.Flock.u64OfLE (Transcript.framed t kind n) ((Transcript.framed t kind n).size - 8)).toNat)) := by
      simp only [squeezeP, StateT.run, h0, if_false]
      rfl
    rw [e, Count.framed_count t kind hn] at h
    cases h
    obtain ⟨fs, hfs, hall⟩ := hp
    refine ⟨qpend_empty, [(Transcript.framed t kind n, n)], by simp, fun r hr => ?_⟩
    rw [List.mem_singleton.1 hr]
    exact ⟨fs, kind, by rw [FlockSoundness.Refine.framed_enc, hfs], hall, hk, h0, hn⟩

theorem sampleP_quiet : quietS.R (sampleP recAsks) (sampleP recAsks) := by
  unfold sampleP
  exact quietS.bind (squeezeP_quiet 1 1 (.inr ⟨rfl, rfl⟩) (by decide)) fun _ => quietS.pure _

theorem sampleVecP_quiet (n : ℕ) (hn : n < 2 ^ 64) : quietS.R (sampleVecP recAsks n) (sampleVecP recAsks n) :=
  squeezeP_quiet 2 n (.inl rfl) hn

/-! ## The algebraic phase -/

theorem zerocheckFw_quiet (m : ℕ) (digest cap root : ByteArray) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128)
    (hm : m < 2 ^ 64) (hd : digest.size < 2 ^ 64) (hc : cap.size < 2 ^ 64) (hr : root.size < 2 ^ 64) :
    quietS.R (zerocheckFw recAsks m digest cap root lay mk) (zerocheckFw recAsks m digest cap root lay mk) := by
  have h13 : m - 13 < 2 ^ 64 := by omega
  unfold zerocheckFw
  sim_walk [label_quiet "flock-r1cs-v0" (by decide) (by decide),
    label_quiet "flock-zerocheck-v0" (by decide) (by decide),
    label_quiet _root_.Flock.Zk.LABEL_PADS (by decide) (by decide),
    absorbBytes_quiet digest hd, absorbBytes_quiet cap hc, absorbBytes_quiet root hr,
    sampleVecP_quiet 6 (by decide), sampleVecP_quiet (m - 13) h13, sampleP_quiet, absorbF_quiet,
    absorbSlice_quiet64]

theorem lincheckFw_quiet (c : _root_.Flock.CircuitFold) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128)
    (zc : _root_.Flock.ZcOut) :
    quietS.R (lincheckFw recAsks c lay mk zc) (lincheckFw recAsks c lay mk zc) := by
  unfold lincheckFw
  sim_walk [label_quiet "flock-lincheck-v0" (by decide) (by decide), sampleP_quiet, absorbF_quiet,
    absorbSlice_quiet64]

theorem repPiop_quiet (st : _root_.Flock.Setup) (lay : _root_.Flock.Zk.PadLayout) (cap root : ByteArray)
    (extras : Array _root_.Flock.Claim) (mk : ℕ → F128) (hm : st.m < 2 ^ 64) (hd : st.digest.size < 2 ^ 64)
    (hc : cap.size < 2 ^ 64) (hr : root.size < 2 ^ 64) :
    quietS.R (repPiop recAsks st lay cap root extras mk) (repPiop recAsks st lay cap root extras mk) := by
  unfold repPiop
  sim_walk [zerocheckFw_quiet st.m st.digest cap root lay mk hm hd hc hr, lincheckFw_quiet]

theorem quiet_run' {α : Type} {p : TMW RecM α} (hp : quietS.R p p) {t : Transcript}
    {s : Array (ByteArray × ℕ) × ℕ} {a : α} {s' : Array (ByteArray × ℕ) × ℕ}
    (h : (p.run' t).run s = .ok (a, s')) (ht : QPend t.pending) :
    ∃ new : List (ByteArray × ℕ), s'.1.toList = s.1.toList ++ new ∧ ∀ r ∈ new, QRound r := by
  rw [StateT.run'_eq, StateT.run_map] at h
  rcases hx : (p.run t).run s with _ | ⟨⟨a', t'⟩, s''⟩
  · rw [hx] at h; cases h
  · rw [hx] at h
    cases h
    exact (hp t s a' t' s'' hx ht).2

/-- **The algebraic phase's scheduled rounds are quiet past the first**: where `Skeleton.rounds` returns, each round
but the zerocheck's head is plain frames and a squeeze of its count. -/
theorem rounds_quiet {st : _root_.Flock.Setup} {k0 : ℕ} {cap dom : ByteArray} {alg : Array (ByteArray × ℕ)}
    (hm : st.m < 2 ^ 64) (hd : st.digest.size < 2 ^ 64) (hc : cap.size < 2 ^ 64) (hdom : dom.size < 2 ^ 64)
    (h : _root_.Flock.Firewall.Skeleton.rounds st k0 cap dom = .ok alg) :
    ∀ j r, 1 ≤ j → alg[j]? = some r → QRound r := by
  have hT := quietS.bind (repPiop_quiet st (_root_.Flock.Zk.PadLayout.new st.m st.kLog k0) cap
    (ByteArray.mk (Array.replicate 64 0)) #[] (fun _ => 0) hm hd hc
    (show (Array.replicate 64 (0 : UInt8)).size < 2 ^ 64 by simp)) fun _ => quietS.pure ()
  have h0 : QPend (Transcript.init #[] dom).pending :=
    ⟨[.dom dom], by rw [FlockSoundness.Refine.init_enc]; simp [encs], fun f hf => by
      rw [List.mem_singleton.1 hf]; exact ⟨hdom, fun b e => by cases e⟩⟩
  unfold _root_.Flock.Firewall.Skeleton.rounds at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  · rename_i v hv
    obtain ⟨_, rs, k⟩ := v
    obtain ⟨new, hnew, hq⟩ := quiet_run' (p := _) hT hv h0
    simp only [List.nil_append, Array.toList_empty] at hnew
    split at h
    · split at h
      · cases h
      · cases h
        intro j r hj hr
        rw [Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds_ne (by omega)] at hr
        obtain ⟨hlt, hget⟩ := Array.getElem?_eq_some_iff.1 hr
        refine hq r ?_
        rw [← hnew, ← hget]
        exact Array.mem_toList_iff.2 (Array.getElem_mem hlt)
    · cases h

end Claims.ZKServer.Quiet
