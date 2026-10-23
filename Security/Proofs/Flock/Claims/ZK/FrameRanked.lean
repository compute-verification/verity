import Proofs.Flock.Claims.ZK.FrameOuter
import Proofs.Flock.Claims.ZK.FramePoints

/-!
# The frames refuse a short rank (`FrameUnrefused`'s rank fault)

At a `Live` statement the loop serves, on a live history of `refK x` openings whose last coins the frame takes, no
table's rep shows a rank short of full over its mask slot's words (`Masked.Thrown`'s first disjunct, `RankBad`): the
rep's frame runs the rank check at the opening's label (`Masker.label`), on the algebraic phase's coins as the stream
answered them, after the earlier rep's claim points.

* `Ranked.Good`, `label_good`, `answer_good`: before the opening's label the frame is in the algebraic phase, outside
  level 0 and the inner proof; once past it (`done`), it is in a masked phase only at level 0, and the rank check held
  at its coins and earlier reps' points (`RankHeld`). Every label keeps that, and so does every answer, which before
  `done` keeps the coins and after it leaves them as they were.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec Transcript TMW)
open _root_.Flock.Firewall.Frame (Cfg Masker MaskSlot K_SKIP isLabel LABEL_PCS LABEL_ZC maskRank claimPoints)
open FlockSoundness.Refine (Frame encs)

namespace Ranked

/-- **The rank check held** at the frame's coins and earlier reps' claim points, at the rep's mask slot if it has one. -/
def RankHeld (c : Cfg) (s : Masker) : Prop :=
  ∀ ms, c.mask = some ms → ∃ P, claimPoints c.lay s.coins = .ok P ∧
    maskRank (s.prior ++ P) ms (c.lay.nLc + K_SKIP) = 128 * (s.prior ++ P).size

/-- **The frame's phase and rank**: before the opening's label (`done`) it is in the algebraic phase, outside level 0
and the inner proof; past it, in a masked phase only at level 0, with the rank check held. -/
def Good (c : Cfg) (s : Masker) : Prop :=
  (s.done = false → s.on = true ∧ s.l0 = false ∧ s.inner = none) ∧
    (s.done = true → (s.on = true → s.l0 = true) ∧ RankHeld c s)

/-- Two states alike in what `Good` reads. -/
def Same (s s' : Masker) : Prop :=
  s'.done = s.done ∧ s'.on = s.on ∧ s'.l0 = s.l0 ∧ s'.inner = s.inner ∧ s'.coins = s.coins ∧ s'.prior = s.prior

theorem good_of_same {c : Cfg} {s s' : Masker} (hg : Good c s) (h : Same s s') : Good c s' := by
  obtain ⟨hd, hon, hl, hi, hc, hp⟩ := h
  obtain ⟨h1, h2⟩ := hg
  refine ⟨fun h' => ?_, fun h' => ?_⟩
  · rw [hd] at h'
    obtain ⟨a, b, c'⟩ := h1 h'
    exact ⟨hon ▸ a, hl ▸ b, hi ▸ c'⟩
  · rw [hd] at h'
    obtain ⟨a, b⟩ := h2 h'
    refine ⟨fun h'' => hl ▸ a (hon ▸ h''), fun ms hms => ?_⟩
    rw [hc, hp]
    exact b ms hms

theorem same_refl (s : Masker) : Same s s := ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem same_trans {s s' s'' : Masker} (h : Same s s') (h' : Same s' s'') : Same s s'' := by
  obtain ⟨a1, a2, a3, a4, a5, a6⟩ := h
  obtain ⟨b1, b2, b3, b4, b5, b6⟩ := h'
  exact ⟨b1.trans a1, b2.trans a2, b3.trans a3, b4.trans a4, b5.trans a5, b6.trans a6⟩

theorem eq_of_isLabel {l : ByteArray} {L : String} (h : isLabel l L = true) : l = L.toUTF8 := by
  unfold isLabel at h
  have e := beq_iff_eq.1 h
  cases l
  cases h' : L.toUTF8
  rw [h'] at e
  cases e
  rfl

theorem isLabel_self (L : String) : isLabel L.toByteArray L = true := by
  unfold isLabel
  exact beq_self_eq_true _

theorem pcs_other : isLabel LABEL_PCS.toByteArray _root_.Flock.Zk.LABEL_CLOSE = false ∧
    isLabel LABEL_PCS.toByteArray _root_.Flock.Zk.LABEL_INNER = false ∧ isLabel LABEL_PCS.toByteArray LABEL_ZC = false ∧
    isLabel LABEL_PCS.toByteArray _root_.Flock.Zk.LABEL_OPEN = false := ⟨rfl, rfl, rfl, rfl⟩

theorem close_other : isLabel _root_.Flock.Zk.LABEL_CLOSE.toByteArray LABEL_PCS = false ∧
    isLabel _root_.Flock.Zk.LABEL_CLOSE.toByteArray _root_.Flock.Zk.LABEL_INNER = false ∧
    isLabel _root_.Flock.Zk.LABEL_CLOSE.toByteArray LABEL_ZC = false ∧
    isLabel _root_.Flock.Zk.LABEL_CLOSE.toByteArray _root_.Flock.Zk.LABEL_OPEN = false := ⟨rfl, rfl, rfl, rfl⟩

theorem inner_other : isLabel _root_.Flock.Zk.LABEL_INNER.toByteArray LABEL_PCS = false ∧
    isLabel _root_.Flock.Zk.LABEL_INNER.toByteArray _root_.Flock.Zk.LABEL_CLOSE = false ∧
    isLabel _root_.Flock.Zk.LABEL_INNER.toByteArray LABEL_ZC = false ∧
    isLabel _root_.Flock.Zk.LABEL_INNER.toByteArray _root_.Flock.Zk.LABEL_OPEN = false := ⟨rfl, rfl, rfl, rfl⟩

theorem zc_other : isLabel LABEL_ZC.toByteArray LABEL_PCS = false ∧
    isLabel LABEL_ZC.toByteArray _root_.Flock.Zk.LABEL_CLOSE = false ∧
    isLabel LABEL_ZC.toByteArray _root_.Flock.Zk.LABEL_INNER = false ∧
    isLabel LABEL_ZC.toByteArray _root_.Flock.Zk.LABEL_OPEN = false := ⟨rfl, rfl, rfl, rfl⟩

theorem open_other : isLabel _root_.Flock.Zk.LABEL_OPEN.toByteArray LABEL_PCS = false ∧
    isLabel _root_.Flock.Zk.LABEL_OPEN.toByteArray _root_.Flock.Zk.LABEL_CLOSE = false ∧
    isLabel _root_.Flock.Zk.LABEL_OPEN.toByteArray _root_.Flock.Zk.LABEL_INNER = false ∧
    isLabel _root_.Flock.Zk.LABEL_OPEN.toByteArray LABEL_ZC = false := ⟨rfl, rfl, rfl, rfl⟩

theorem rankHeld_of {c : Cfg} {s s' : Masker} (h : RankHeld c s) (hc : s'.coins = s.coins) (hp : s'.prior = s.prior) :
    RankHeld c s' := by
  intro ms hms
  rw [hc, hp]
  exact h ms hms

/-- **A label keeps the frame's phase and rank**, its coins and earlier reps' points. -/
theorem label_good {c : Cfg} {s : Masker} {l : ByteArray} {q : Masker × ByteArray} (hg : Good c s)
    (h : s.label c l = .ok q) : Good c q.1 ∧ q.1.coins = s.coins ∧ q.1.prior = s.prior := by
  obtain ⟨g1, g2⟩ := hg
  by_cases hP : isLabel l LABEL_PCS = true
  · rw [eq_of_isLabel hP] at h
    cases hon : s.on
    · have e : s.label c LABEL_PCS.toUTF8 = .ok (s, _root_.Flock.Firewall.Frame.labelFrame LABEL_PCS.toUTF8) := by
        simp [Masker.label, hon]
        rfl
      rw [e] at h
      cases h
      exact ⟨⟨g1, g2⟩, rfl, rfl⟩
    · obtain ⟨p1, p2, p3, p4⟩ := pcs_other
      by_cases hpos : s.pos = c.nPiop
      · cases hm : c.mask with
        | none =>
          simp [Masker.label, hon, hm, isLabel_self, p1, p2, p3, p4, hpos] at h
          cases h
          exact ⟨⟨fun hd => by simp at hd, fun _ => ⟨fun h' => by simp at h', fun ms hms => by simp [hm] at hms⟩⟩,
            rfl, rfl⟩
        | some ms =>
          cases hcp : claimPoints c.lay s.coins with
          | error e =>
            simp [Masker.label, hon, hm, isLabel_self, p1, p2, p3, p4, hpos, hcp] at h
            cases h
          | ok P =>
            simp [Masker.label, hon, hm, isLabel_self, p1, p2, p3, p4, hpos, hcp] at h
            rw [show ∀ f : Array (Array F128) → V (Masker × ByteArray), (Except.ok P >>= f) = f P from
              fun _ => rfl] at h
            split at h
            · rename_i hr
              cases h
              refine ⟨⟨fun hd => by simp at hd, fun _ => ⟨fun h' => by simp at h', fun ms' hms' => ?_⟩⟩, rfl, rfl⟩
              rw [hm] at hms'
              cases hms'
              exact ⟨P, hcp, by rw [Array.size_append]; exact hr⟩
            · cases h
      · cases hm : c.mask <;> simp [Masker.label, hon, hm, isLabel_self, p1, p2, p3, p4, hpos] at h <;> cases h
  by_cases hC : isLabel l _root_.Flock.Zk.LABEL_CLOSE = true
  · rw [eq_of_isLabel hC] at h
    obtain ⟨p1, p2, p3, p4⟩ := close_other
    cases hl0 : s.l0
    · simp [Masker.label, isLabel_self, p1, p2, p3, p4, hl0] at h
      cases h
      exact ⟨⟨g1, g2⟩, rfl, rfl⟩
    · by_cases hpos : s.pos = c.order.size
      · by_cases hrho : s.rho.size = _root_.Flock.Firewall.Frame.RHO_COINS
        · simp [Masker.label, isLabel_self, p1, p2, p3, p4, hl0, hpos, hrho] at h
          cases h
          refine ⟨⟨fun hd => absurd (g1 hd).2.1 (by simp [hl0]), fun hd => ⟨fun h' => by simp at h', ?_⟩⟩, rfl, rfl⟩
          exact rankHeld_of (g2 hd).2 rfl rfl
        · simp [Masker.label, isLabel_self, p1, p2, p3, p4, hl0, hpos, hrho] at h
          cases h
      · simp [Masker.label, isLabel_self, p1, p2, p3, p4, hl0, hpos] at h
        cases h
  by_cases hI : isLabel l _root_.Flock.Zk.LABEL_INNER = true
  · rw [eq_of_isLabel hI] at h
    obtain ⟨p1, p2, p3, p4⟩ := inner_other
    cases hi : s.inner with
    | some k =>
      simp [Masker.label, isLabel_self, p1, p2, p3, p4, hi] at h
      cases h
    | none =>
      cases hd : s.done
      · simp [Masker.label, isLabel_self, p1, p2, p3, p4, hi, hd] at h
        cases h
      · cases hon : s.on
        · simp [Masker.label, isLabel_self, p1, p2, p3, p4, hi, hd, hon] at h
          split at h
          · cases h
          · cases h
            refine ⟨⟨fun hd' => by simp [hd] at hd', fun _ => ⟨fun h' => by simp [hon] at h', ?_⟩⟩, rfl, rfl⟩
            exact rankHeld_of (g2 hd).2 rfl rfl
        · simp [Masker.label, isLabel_self, p1, p2, p3, p4, hi, hd, hon] at h
          cases h
  by_cases hZ : isLabel l LABEL_ZC = true
  · rw [eq_of_isLabel hZ] at h
    obtain ⟨p1, p2, p3, p4⟩ := zc_other
    cases hd : s.done
    · simp [Masker.label, isLabel_self, p1, p2, p3, p4, hd] at h
      cases h
      exact ⟨⟨fun _ => ⟨rfl, (g1 hd).2.1, (g1 hd).2.2⟩, fun hd' => by simp [hd] at hd'⟩, rfl, rfl⟩
    · simp [Masker.label, isLabel_self, p1, p2, p3, p4, hd] at h
      cases h
      exact ⟨⟨g1, g2⟩, rfl, rfl⟩
  by_cases hO : isLabel l _root_.Flock.Zk.LABEL_OPEN = true
  · rw [eq_of_isLabel hO] at h
    obtain ⟨p1, p2, p3, p4⟩ := open_other
    simp [Masker.label, isLabel_self, p1, p2, p3, p4] at h
    split at h
    · rename_i hc
      cases h
      exact ⟨⟨fun hd => by simp [hc.1.1] at hd, fun hd => ⟨fun _ => rfl, rankHeld_of (g2 hd).2 rfl rfl⟩⟩, rfl, rfl⟩
    · cases h
      exact ⟨⟨g1, g2⟩, rfl, rfl⟩
  simp only [Bool.not_eq_true] at hP hC hI hZ hO
  rw [label_plain c s l ⟨hP, hC, hI, hZ, hO⟩] at h
  cases h
  exact ⟨⟨g1, g2⟩, rfl, rfl⟩

end Ranked

end Claims.ZKServer
