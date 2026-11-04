module

public import Proofs.Flock.SessionLoop.LiveInner
public import Proofs.Flock.Claims.ZK.L0PinSwitch
import all Init.Data.String.Defs
import all Init.Data.Array.Basic

/-!
# The frame at a rep's ring-switch rounds

From the opening's label on, the frame frees no word of a round and releases it as it came:

* `Loop.label_pcs_out`: the opening's label, where the frame takes it, in the algebraic phase or past it, leaves the
  frame outside every masked phase and releases its label frame.
* `Loop.roundFree_pcsAt`, `Loop.check_pcsAt`: a round of the opening's label and then plain frames frees no word and
  leaves as its frames came, so the frame's check passes a device round there only where it is the scheduled round.
* `Loop.check_rs`: at the ring switch's rounds (`Switch.rsBody` from empty pending bytes), the first at any state and
  the later ones outside every masked phase, the check passes only the scheduled body and releases it, which holds no
  domain frame (`Loop.dropDom_rs`).
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128)
open _root_.Flock.Firewall.Frame (Cfg Masker roundFree check labelFrame LABEL_PCS claimPoints)
open FlockSoundness.Refine (Frame encs)

namespace Loop

/-- The opening's label outside every masked phase leaves the frame as it is and releases its label frame. -/
theorem label_pcs_off (c : Cfg) (s : Masker) (hon : s.on = false) :
    s.label c LABEL_PCS.toUTF8 = .ok (s, labelFrame LABEL_PCS.toUTF8) := by
  obtain ⟨p1, p2, p3, p4⟩ := Ranked.pcs_other
  simp [Masker.label, hon, Ranked.isLabel_self, p1, p2, p3, p4]
  rfl

/-- **The opening's label**, where the frame takes it, leaves it outside every masked phase and releases its label
frame. -/
theorem label_pcs_out (c : Cfg) (s : Masker) {q : Masker × ByteArray} (h : s.label c LABEL_PCS.toUTF8 = .ok q) :
    q.1.on = false ∧ q.2 = labelFrame LABEL_PCS.toUTF8 := by
  cases hon : s.on with
  | false =>
    rw [label_pcs_off c s hon] at h
    cases h
    exact ⟨hon, rfl⟩
  | true =>
    obtain ⟨p1, p2, p3, p4⟩ := Ranked.pcs_other
    by_cases hpos : s.pos = c.nPiop
    · cases hm : c.mask with
      | none =>
        simp [Masker.label, hon, hm, Ranked.isLabel_self, p1, p2, p3, p4, hpos] at h
        cases h
        exact ⟨rfl, rfl⟩
      | some ms =>
        cases hcp : claimPoints c.lay s.coins with
        | error e =>
          simp [Masker.label, hon, hm, Ranked.isLabel_self, p1, p2, p3, p4, hpos, hcp] at h
          cases h
        | ok P =>
          simp [Masker.label, hon, hm, Ranked.isLabel_self, p1, p2, p3, p4, hpos, hcp] at h
          rw [show ∀ f : Array (Array F128) → V (Masker × ByteArray), (Except.ok P >>= f) = f P from
            fun _ => rfl] at h
          split at h
          · cases h
            exact ⟨rfl, rfl⟩
          · cases h
    · cases hm : c.mask <;> simp [Masker.label, hon, hm, Ranked.isLabel_self, p1, p2, p3, p4, hpos] at h <;> cases h

/-- **A round from the opening's label**: where the frame takes the opening's label and then plain frames, in the
algebraic phase or past it, it frees no word and releases the frames as they came, then the squeeze. -/
theorem roundFree_pcsAt {c : Cfg} {s : Masker} {fs : List Frame} {k : UInt8} {n : ℕ}
    (hm : ∀ f ∈ fs, midFrame f) (hp : ∀ f ∈ fs, plainFrame f) (hk : k = 2 ∨ (k = 1 ∧ n = 1))
    (h0 : ¬ (n = 0 ∧ k = 2)) (hn : n < 2 ^ 64) {q : ByteArray × Masker × Array ℕ}
    (h : roundFree c s (ByteArray.mk (encs (.label LABEL_PCS.toUTF8 :: fs) ++ (Frame.sq k n).enc).toArray) n =
      .ok q) :
    q.2.2 = #[] ∧
      q.1.data.toList = encs (maskFrames (fun _ v => v) 0 (.label LABEL_PCS.toUTF8 :: fs)) ++ (Frame.sq k n).enc := by
  have hs := roundFree_ok_inner h
  have hm' : ∀ f ∈ Frame.label LABEL_PCS.toUTF8 :: fs, midFrame f := by
    intro f hf
    rcases List.mem_cons.1 hf with rfl | hf
    · show LABEL_PCS.toUTF8.size < 2 ^ 64
      decide
    · exact hm f hf
  rw [roundFree_encs c s _ k n hs hm' hk h0 hn] at h
  obtain ⟨x, hx, h⟩ := bindV_ok h
  rw [List.foldlM_cons] at hx
  obtain ⟨y, hy, hx⟩ := bindV_ok hx
  obtain ⟨ql, hl, hy1, hy2, hy3⟩ : ∃ ql, s.label c LABEL_PCS.toUTF8 = .ok ql ∧ y.1 = ql.1 ∧
      y.2.1 = ByteArray.empty ++ ql.2 ∧ y.2.2.1 = #[] := by
    simp only [stepAt] at hy
    cases hl : s.label c LABEL_PCS.toUTF8 with
    | error e => rw [hl] at hy; cases hy
    | ok ql =>
      rw [hl] at hy
      cases hy
      exact ⟨ql, rfl, rfl, rfl, rfl⟩
  obtain ⟨hof, hout⟩ := label_pcs_out c s hl
  obtain ⟨y1, y2, y3, y4⟩ := y
  simp only at hy1 hy2 hy3
  subst hy2 hy3
  rw [← hy1] at hof
  obtain ⟨r, hr, -, h2⟩ := foldlM_stepAt_off c fs y1 (ByteArray.empty ++ ql.2) #[] y4 0 hof hp
  have hfr := foldlM_stepAt_off_fr c fs y1 (ByteArray.empty ++ ql.2) #[] y4 r hof hp hr
  have hxr : x = r := Except.ok.inj (hx.symm.trans hr)
  rw [hxr] at h
  obtain ⟨m, o, fr, q'⟩ := r
  simp only at h2 hfr
  subst hfr
  have e := exMap_ok_eq h
  subst e
  refine ⟨rfl, ?_⟩
  dsimp only
  rw [data_append', h2, data_append', hout, labelFrame_data, sq_enc]
  simp [maskFrames, encs]

/-- **From the opening's label a round passes only as itself**: where the frame's check passes the device round `v`
against a scheduled round of the opening's label and then plain frames, at any state, `v` is that round and leaves as
its frames. -/
theorem check_pcsAt {c : Cfg} {m s' : Masker} {fs : List Frame} {k : UInt8} {n : ℕ} {v out : ByteArray}
    (hm : ∀ f ∈ fs, midFrame f) (hp : ∀ f ∈ fs, plainFrame f) (hk : k = 2 ∨ (k = 1 ∧ n = 1))
    (h0 : ¬ (n = 0 ∧ k = 2)) (hn : n < 2 ^ 64)
    (h : check c m v (ByteArray.mk (encs (.label LABEL_PCS.toUTF8 :: fs) ++ (Frame.sq k n).enc).toArray) n =
      .ok (out, s', none)) :
    v = ByteArray.mk (encs (.label LABEL_PCS.toUTF8 :: fs) ++ (Frame.sq k n).enc).toArray ∧
      out.data.toList = encs (maskFrames (fun _ v => v) 0 (.label LABEL_PCS.toUTF8 :: fs)) ++ (Frame.sq k n).enc := by
  obtain ⟨q, hq, hsz, hd⟩ := passes_none (check_none_passes h)
  rw [(roundFree_pcsAt hm hp hk h0 hn hq).1] at hd
  have hv := eq_of_firstDiff hsz hd
  refine ⟨hv, ?_⟩
  obtain ⟨fr, hf⟩ := Sched.roundFree_round (check_none h)
  rw [hv] at hf
  exact (roundFree_pcsAt hm hp hk h0 hn hf).2

/-! ## The ring switch's rounds -/

/-- The frames of a ring-switch claim's round: the ring switch's label and the claim's switched vector. -/
def rsFrames (vs : Array F128) : List Frame := [.label "flock-ring-switch-v0".toUTF8, .vec vs.toList]

theorem rsFrames_mid (vs : Array F128) (hs : vs.size < 2 ^ 64) : ∀ f ∈ rsFrames vs, midFrame f := by
  intro f hf
  simp only [rsFrames, List.mem_cons, List.not_mem_nil, or_false] at hf
  rcases hf with rfl | rfl
  · show "flock-ring-switch-v0".toUTF8.size < 2 ^ 64
    decide
  · show vs.toList.length < 2 ^ 64
    simpa using hs

theorem rsFrames_plain (vs : Array F128) : ∀ f ∈ rsFrames vs, plainFrame f := by
  intro f hf
  simp only [rsFrames, List.mem_cons, List.not_mem_nil, or_false] at hf
  rcases hf with rfl | rfl
  · show plainLabel "flock-ring-switch-v0".toUTF8; unfold plainLabel; decide
  · trivial

theorem maskFrames_rs (vs : Array F128) : maskFrames (fun _ v => v) 0 (rsFrames vs) = rsFrames vs := by
  simp only [rsFrames, maskFrames, maskVals_id]

theorem maskFrames_pcs_rs (vs : Array F128) :
    maskFrames (fun _ v => v) 0 (.label LABEL_PCS.toUTF8 :: rsFrames vs) = .label LABEL_PCS.toUTF8 :: rsFrames vs := by
  simp only [rsFrames, maskFrames, maskVals_id]

theorem rsBody_zero (sv : ℕ → Array F128) {N : ℕ} (hN : 0 < N) :
    Switch.rsBody sv N [] 0 = encs (.label LABEL_PCS.toUTF8 :: rsFrames (sv 0)) ++ (Frame.sq 2 7).enc := by
  rw [Switch.rsBody, if_pos hN, if_pos rfl]
  simp only [rsFrames, LABEL_PCS, encs, List.flatMap_cons, List.flatMap_nil, List.append_nil, List.nil_append,
    List.append_assoc]

theorem rsBody_mid (sv : ℕ → Array F128) {N k : ℕ} (h0 : k ≠ 0) (hk : k < N) :
    Switch.rsBody sv N [] k = encs (rsFrames (sv k)) ++ (Frame.sq 2 7).enc := by
  rw [Switch.rsBody, if_pos hk, if_neg h0, List.nil_append, rsFrames]

theorem rsBody_last (sv : ℕ → Array F128) (N : ℕ) : Switch.rsBody sv N [] N = encs [] ++ (Frame.sq 2 N).enc := by
  rw [Switch.rsBody, if_neg (lt_irrefl _)]
  rfl

/-- **The frame at the ring switch's rounds**: where the frame's check passes the device round `v` against the ring
switch's round `k ≤ N` from empty pending bytes, of `7` coins below `N` and `N` at `N`, the first at any state and the
later ones outside every masked phase, the round releases as the scheduled body. -/
theorem check_rs {c : Cfg} {m s' : Masker} {sv : ℕ → Array F128} {N k n : ℕ} {v out : ByteArray}
    (hon : k ≠ 0 → m.on = false) (hkN : k ≤ N) (hN : N ≠ 0) (hN' : N < 2 ^ 64)
    (hs : k < N → (sv k).size < 2 ^ 64) (hn : n = if k < N then 7 else N)
    (h : check c m v (ByteArray.mk (Switch.rsBody sv N [] k).toArray) n = .ok (out, s', none)) :
    out.data.toList = Switch.rsBody sv N [] k := by
  by_cases hk : k < N
  · rw [if_pos hk] at hn
    subst hn
    by_cases h0 : k = 0
    · subst h0
      rw [rsBody_zero sv hk] at h ⊢
      rw [(check_pcsAt (rsFrames_mid _ (hs hk)) (rsFrames_plain _) (.inl rfl) (by decide) (by decide) h).2,
        maskFrames_pcs_rs]
    · rw [rsBody_mid sv h0 hk] at h ⊢
      rw [(check_offAt (hon h0) (rsFrames_mid _ (hs hk)) (rsFrames_plain _) (.inl rfl) (by decide) (by decide)
        h).2, maskFrames_rs]
  · rw [if_neg hk] at hn
    subst n
    obtain rfl : k = N := by omega
    rw [rsBody_last sv _] at h ⊢
    rw [(check_offAt (fs := []) (hon (by omega)) (fun _ h => nomatch h) (fun _ h => nomatch h) (.inl rfl)
      (fun h' => hN h'.1) hN' h).2]
    rfl

theorem rsBody_noDom (sv : ℕ → Array F128) {N k : ℕ} (hkN : k ≤ N) :
    ∃ M kk nn, (∀ f ∈ M, noDom f) ∧ Switch.rsBody sv N [] k = encs M ++ (Frame.sq kk nn).enc := by
  by_cases hk : k < N
  · by_cases h0 : k = 0
    · subst h0
      refine ⟨_, _, _, fun f hf => ?_, rsBody_zero sv hk⟩
      simp only [rsFrames, List.mem_cons, List.not_mem_nil, or_false] at hf
      rcases hf with rfl | rfl | rfl <;> trivial
    · refine ⟨_, _, _, fun f hf => ?_, rsBody_mid sv h0 hk⟩
      simp only [rsFrames, List.mem_cons, List.not_mem_nil, or_false] at hf
      rcases hf with rfl | rfl <;> trivial
  · obtain rfl : k = N := by omega
    exact ⟨[], _, _, (fun _ h => nomatch h), rsBody_last sv _⟩

/-- **The ring switch's rounds hold no domain frame**: their message is as it is without one. -/
theorem dropDom_rs (sv : ℕ → Array F128) {N k : ℕ} (hkN : k ≤ N) :
    dropDom (0 :: Switch.rsBody sv N [] k) = 0 :: Switch.rsBody sv N [] k := by
  obtain ⟨M, kk, nn, hM, he⟩ := rsBody_noDom sv hkN
  rw [he]
  exact dropDom_noDom M kk nn hM

end Loop

end Claims.ZKServer
