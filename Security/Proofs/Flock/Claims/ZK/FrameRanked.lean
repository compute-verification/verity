import Proofs.Flock.Claims.ZK.FrameOuter
import Proofs.Flock.Claims.ZK.FramePoints

/-!
# The frame's rank check along a rep's requests (`FrameUnrefused`'s rank fault, at the frame)

A rep's frame runs the rank check at the opening's label (`Masker.label`), on the algebraic phase's coins as the stream
answered them, after the earlier rep's claim points. Along the loop's requests for a rep:

* `Ranked.Good`: before the opening's label (`done`) the frame is in the algebraic phase, outside level 0 and the inner
  proof; past it, in a masked phase only at level 0, and the rank check held at its coins and the earlier reps' points
  (`RankHeld`). Every label keeps that (`label_good`), and so does every round (`roundFree_keep`, `step_round`) and
  every answer (`answer_keep`), which before the label appends the round's coins and after it keeps the coins.
* `Ranked.Inv`, `run_head`: from a rep's start past its first round (the zerocheck's head) and then each round's coins
  and the next (`pairs`), the frame holds rounds `0 … d − 1`'s coins, each answered the round's count, `d` every round
  answered before the opening's label.
* **`Ranked.ownReqs_rank`**: so where a rep's requests up to an inner proof's round (`ownReqs`) run on into the inner
  proof, the rank check held at the rounds' coins, after the earlier rep's claim points (`run_prior_eq`).
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec Transcript TMW)
open _root_.Flock.Firewall.Frame (Cfg Masker MaskSlot K_SKIP isLabel LABEL_PCS LABEL_ZC maskRank claimPoints)
open FlockSoundness.Refine (Frame encs)

namespace Ranked

open Loop

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
    (h : s.label c l = .ok q) :
    Good c q.1 ∧ q.1.coins = s.coins ∧ q.1.prior = s.prior ∧ (s.done = true → q.1.done = true) := by
  obtain ⟨g1, g2⟩ := hg
  by_cases hP : isLabel l LABEL_PCS = true
  · rw [eq_of_isLabel hP] at h
    cases hon : s.on
    · have e : s.label c LABEL_PCS.toUTF8 = .ok (s, _root_.Flock.Firewall.Frame.labelFrame LABEL_PCS.toUTF8) := by
        simp [Masker.label, hon]
        rfl
      rw [e] at h
      cases h
      exact ⟨⟨g1, g2⟩, rfl, rfl, id⟩
    · obtain ⟨p1, p2, p3, p4⟩ := pcs_other
      by_cases hpos : s.pos = c.nPiop
      · cases hm : c.mask with
        | none =>
          simp [Masker.label, hon, hm, isLabel_self, p1, p2, p3, p4, hpos] at h
          cases h
          exact ⟨⟨fun hd => by simp at hd, fun _ => ⟨fun h' => by simp at h', fun ms hms => by simp [hm] at hms⟩⟩,
            rfl, rfl, fun _ => rfl⟩
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
              refine ⟨⟨fun hd => by simp at hd, fun _ => ⟨fun h' => by simp at h', fun ms' hms' => ?_⟩⟩, rfl, rfl,
                fun _ => rfl⟩
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
      exact ⟨⟨g1, g2⟩, rfl, rfl, id⟩
    · by_cases hpos : s.pos = c.order.size
      · by_cases hrho : s.rho.size = _root_.Flock.Firewall.Frame.RHO_COINS
        · simp [Masker.label, isLabel_self, p1, p2, p3, p4, hl0, hpos, hrho] at h
          cases h
          refine ⟨⟨fun hd => absurd (g1 hd).2.1 (by simp [hl0]), fun hd => ⟨fun h' => by simp at h', ?_⟩⟩, rfl, rfl, id⟩
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
            refine ⟨⟨fun hd' => by simp at hd', fun _ => ⟨fun h' => by simp at h', ?_⟩⟩, rfl, rfl, id⟩
            exact rankHeld_of (g2 hd).2 rfl rfl
        · simp [Masker.label, isLabel_self, p1, p2, p3, p4, hi, hd, hon] at h
          cases h
  by_cases hZ : isLabel l LABEL_ZC = true
  · rw [eq_of_isLabel hZ] at h
    obtain ⟨p1, p2, p3, p4⟩ := zc_other
    cases hd : s.done
    · simp [Masker.label, isLabel_self, p1, p2, p3, p4, hd] at h
      cases h
      exact ⟨⟨fun _ => ⟨rfl, (g1 hd).2.1, (g1 hd).2.2⟩, fun hd' => by simp at hd'⟩, rfl, rfl, id⟩
    · simp [Masker.label, isLabel_self, p1, p2, p3, p4, hd] at h
      cases h
      exact ⟨⟨g1, g2⟩, rfl, rfl, fun _ => hd⟩
  by_cases hO : isLabel l _root_.Flock.Zk.LABEL_OPEN = true
  · rw [eq_of_isLabel hO] at h
    obtain ⟨p1, p2, p3, p4⟩ := open_other
    simp [Masker.label, isLabel_self, p1, p2, p3, p4] at h
    split at h
    · rename_i hc
      cases h
      exact ⟨⟨fun hd => by simp [hc.1.1] at hd, fun hd => ⟨fun _ => rfl, rankHeld_of (g2 hd).2 rfl rfl⟩⟩, rfl, rfl, id⟩
    · cases h
      exact ⟨⟨g1, g2⟩, rfl, rfl, id⟩
  simp only [Bool.not_eq_true] at hP hC hI hZ hO
  rw [label_plain c s l ⟨hP, hC, hI, hZ, hO⟩] at h
  cases h
  exact ⟨⟨g1, g2⟩, rfl, rfl, id⟩

/-- A state `s'` keeps `s`'s phase and rank: `Good`, at `s`'s coins and earlier reps' points. -/
def Keep (c : Cfg) (s s' : Masker) : Prop :=
  Good c s' ∧ s'.coins = s.coins ∧ s'.prior = s.prior ∧ (s.done = true → s'.done = true)

theorem keep_same {c : Cfg} {s m m' : Masker} (hk : Keep c s m) (h : Same m m') : Keep c s m' :=
  ⟨good_of_same hk.1 h, h.2.2.2.2.1.trans hk.2.1, h.2.2.2.2.2.trans hk.2.2.1, fun hd => h.1.trans (hk.2.2.2 hd)⟩

theorem keep_label {c : Cfg} {s m : Masker} {l : ByteArray} {q : Masker × ByteArray} (hk : Keep c s m)
    (h : m.label c l = .ok q) : Keep c s q.1 := by
  obtain ⟨g, hc, hp, hd⟩ := label_good hk.1 h
  exact ⟨g, hc.trans hk.2.1, hp.trans hk.2.2.1, fun h' => hd (hk.2.2.2 h')⟩

theorem absorb_same {c : Cfg} {s : Masker} {v : F128} {q : Masker × F128} (h : s.absorb c v = .ok q) : Same s q.1 := by
  unfold Masker.absorb at h
  cases hon : s.on
  · simp only [hon, Bool.not_false, ↓reduceIte, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact same_refl s
  · simp only [hon, Bool.not_true, Bool.false_eq_true, ↓reduceIte] at h
    cases hi : c.order[s.pos]? with
    | none => simp only [hi] at h; cases h
    | some i =>
      cases hp : c.h[i]? with
      | none => simp only [hi, hp] at h; cases h
      | some p =>
        simp only [hi, hp, pure, Except.pure, Except.ok.injEq] at h
        subst h
        exact ⟨rfl, hon.symm, rfl, rfl, rfl, rfl⟩

theorem forIn_inv {τ : Type} (Q : τ → Prop) (g : ℕ → τ → V (ForInStep τ))
    (hg : ∀ i x y, Q x → g i x = .ok y → Q (match y with | .done b => b | .yield b => b)) :
    ∀ (l : List ℕ) (x y : τ), Q x → forIn l x g = .ok y → Q y
  | [], x, y, hx, h => by cases h; exact hx
  | a :: l, x, y, hx, h => by
    rw [List.forIn_cons] at h
    cases hga : g a x with
    | error e => rw [hga] at h; cases h
    | ok st =>
      rw [hga] at h
      cases st with
      | done b => cases h; exact hg a x _ hx hga
      | yield b => exact forIn_inv Q g hg l b y (hg a x _ hx hga) h

theorem ite_bind_ok {α β : Type} {p : Prop} [Decidable p] {e : String} {a : α} {f : α → V β} {y : β}
    (h : ((if p then throw e else pure a) >>= f) = .ok y) : f a = .ok y := by
  by_cases hp : p
  · rw [ite_eq_left hp] at h
    cases h
  · rw [ite_eq_right hp] at h
    exact h

theorem squeeze_pos {len n : ℕ} {kind : UInt8} (h5 : ¬ (len == 0 && kind == 2) = true) (h6 : ¬ (len != n) = true)
    (h7 : ¬ (!(kind == 2 || kind == 1 && n == 1)) = true) : 1 ≤ n := by
  rcases Nat.eq_zero_or_pos n with rfl | hp
  · simp at h5 h6 h7
    exact absurd h7 (h5 h6)
  · exact hp

open _root_.Flock.Firewall.Frame (roundFree INNER_ROUNDS) in
/-- **A round keeps the frame's phase and rank**, its coins and earlier reps' points, whatever its bytes; it owes its
coins, at least one. -/
theorem roundFree_keep {c : Cfg} {s : Masker} {content : ByteArray} {n : Nat} {q : ByteArray × Masker × Array Nat}
    (hg : Good c s) (h : roundFree c s content n = .ok q) :
    Keep c s q.2.1 ∧ q.2.1.owed = some n ∧ 1 ≤ n := by
  unfold roundFree at h
  by_cases hI : (s.inner == some INNER_ROUNDS) = true
  · simp only [hI, ↓reduceIte] at h
    cases h
  simp only [hI, Bool.false_eq_true, ↓reduceIte, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at h
  obtain ⟨r, hr, h⟩ := bindV_ok h
  have hQ := forIn_inv (fun x : Option (ByteArray × Masker × Array ℕ) × Masker × ByteArray × Array ℕ × ℕ =>
      Keep c s x.2.1 ∧ ∀ v, x.1 = some v → Keep c s v.2.1 ∧ v.2.1.owed = some n ∧ 1 ≤ n) _ ?step _ _ _
    ⟨⟨hg, rfl, rfl, id⟩, fun _ hv => by cases hv⟩ hr
  · obtain ⟨o, m, out, free, pos⟩ := r
    cases o with
    | none => cases h
    | some v => cases h; exact hQ.2 _ rfl
  · intro i x y hx hy
    obtain ⟨o, m, out, free, pos⟩ := x
    obtain ⟨hk, -⟩ := hx
    dsimp only at hy
    by_cases h1 : pos + 16 > content.size
    · rw [ite_eq_left h1] at hy
      by_cases h2 : (pos == content.size) = true
      · rw [ite_eq_left h2] at hy
        cases hy
      · rw [ite_eq_right h2] at hy
        cases hy
    rw [ite_eq_right h1] at hy
    generalize (content.get! pos).toNat = op at hy
    rcases op with _ | _ | _ | _ | _ | _ | op
    all_goals simp only [] at hy
    · cases hy
    · have hy := ite_bind_ok hy
      cases hy
      exact ⟨hk, fun _ hv => by cases hv⟩
    · have hy := ite_bind_ok hy
      obtain ⟨x, hx, hy⟩ := bindV_ok hy
      cases hy
      exact ⟨keep_label hk hx, fun _ hv => by cases hv⟩
    · have hy := ite_bind_ok hy
      generalize (_root_.Flock.u64OfLE content (pos + 8)).toNat = len at hy
      generalize content.get! (pos + 1) = kind at hy
      by_cases h3 : (!(kind == 2 || kind == 1 && len == 1)) = true
      · rw [ite_eq_left h3] at hy
        cases hy
      rw [ite_eq_right h3] at hy
      obtain ⟨r', hr', hy⟩ := bindV_ok hy
      cases hy
      have hQ' : Keep c s r'.1 := by
        refine forIn_inv (fun x : Masker × ByteArray × Array ℕ => Keep c s x.1) _ ?_ _ _ _ hk hr'
        intro i x y hx hy
        by_cases h8 : (x.1.on && decide (x.1.pos < c.nPiop)) = true
        · rw [ite_eq_left h8] at hy
          obtain ⟨a, ha, hy⟩ := bindV_ok hy
          cases hy
          exact keep_same hx (absorb_same ha)
        · rw [ite_eq_right h8] at hy
          obtain ⟨a, ha, hy⟩ := bindV_ok hy
          cases hy
          exact keep_same hx (absorb_same ha)
      exact ⟨hQ', fun _ hv => by cases hv⟩
    · generalize (_root_.Flock.u64OfLE content (pos + 8)).toNat = len at hy
      generalize content.get! (pos + 1) = kind at hy
      by_cases h5 : (len == 0 && kind == 2) = true
      · rw [ite_eq_left h5] at hy
        cases hy
        exact ⟨hk, fun _ hv => by cases hv⟩
      rw [ite_eq_right h5] at hy
      by_cases h4 : (pos + 16 != content.size) = true
      · rw [ite_eq_left h4] at hy
        cases hy
      rw [ite_eq_right h4] at hy
      by_cases h6 : (len != n) = true
      · rw [ite_eq_left h6] at hy
        cases hy
      rw [ite_eq_right h6] at hy
      by_cases h7 : (!(kind == 2 || kind == 1 && n == 1)) = true
      · rw [ite_eq_left h7] at hy
        cases hy
      rw [ite_eq_right h7] at hy
      have hn := squeeze_pos h5 h6 h7
      have hk' : Keep c s { m with owed := some n } := keep_same hk ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩
      split at hy
      · obtain ⟨u, -, hy⟩ := bindV_ok hy
        cases hy
        exact ⟨hk, fun v hv => by cases hv; exact ⟨hk', rfl, hn⟩⟩
      · cases hy
        exact ⟨hk, fun v hv => by cases hv; exact ⟨hk', rfl, hn⟩⟩
    · have hy := ite_bind_ok hy
      cases hy
      exact ⟨hk, fun _ hv => by cases hv⟩
    · cases hy

theorem good_congr {c : Cfg} {s s' : Masker} (hg : Good c s) (hd : s'.done = s.done) (hon : s'.on = s.on)
    (hl : s'.l0 = s.l0) (hi : s.done = false → s'.inner = s.inner) (hc : s.done = true → s'.coins = s.coins)
    (hp : s'.prior = s.prior) : Good c s' := by
  obtain ⟨g1, g2⟩ := hg
  refine ⟨fun h' => ?_, fun h' => ?_⟩
  · rw [hd] at h'
    obtain ⟨a, b, e⟩ := g1 h'
    exact ⟨hon ▸ a, hl ▸ b, (hi h').trans e⟩
  · rw [hd] at h'
    obtain ⟨a, b⟩ := g2 h'
    exact ⟨fun h'' => hl ▸ a (hon ▸ h''), rankHeld_of b (hc h') hp⟩

/-- Coins the frame takes at level 0 or in the inner proof keep its phase, coins and earlier reps' points. -/
theorem answer_proj {c : Cfg} {s s' : Masker} {cs : Array F128} (hx : s.inner.isSome = true ∨ s.l0 = true)
    (h : s.answer c cs = .ok s') :
    s'.done = s.done ∧ s'.on = s.on ∧ s'.l0 = s.l0 ∧ s'.coins = s.coins ∧ s'.prior = s.prior := by
  unfold Masker.answer at h
  cases ho : s.owed with
  | none => rw [ho] at h; cases h
  | some n =>
    by_cases hc : (cs.size != n) = true
    · simp only [ho, hc, ↓reduceIte] at h
      cases h
    simp only [ho, hc, Bool.false_eq_true, ↓reduceIte] at h
    cases hi : s.inner with
    | some k =>
      simp only [hi] at h
      obtain ⟨u, -, h⟩ := bindV_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      split <;> (try split) <;> exact ⟨rfl, rfl, rfl, rfl, rfl⟩
    | none =>
      have hl : s.l0 = true := by
        rcases hx with hx | hx
        · rw [hi] at hx; cases hx
        · exact hx
      simp only [hi, hl, ↓reduceIte] at h
      by_cases hp : (s.pos != c.order.size) = true
      · simp only [hp, ↓reduceIte, pure, Except.pure, Except.ok.injEq] at h
        subst h
        exact ⟨rfl, rfl, hl.symm, rfl, rfl⟩
      · simp only [hp, Bool.false_eq_true, ↓reduceIte] at h
        obtain ⟨r, -, h⟩ := bindV_ok h
        simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        exact ⟨rfl, rfl, hl.symm, rfl, rfl⟩

/-- **Coins the frame takes keep its phase and rank**: before the opening's label they are the coins so far and the
round's; past it, the coins so far. -/
theorem answer_keep {c : Cfg} {s s' : Masker} {cs : Array F128} (hg : Good c s) (h : s.answer c cs = .ok s') :
    Good c s' ∧ s'.prior = s.prior ∧ s'.done = s.done ∧ (s.done = false → s'.coins = s.coins ++ cs) ∧
      (s.done = true → s'.coins = s.coins) := by
  cases hd : s.done
  · obtain ⟨hon, hl, hi⟩ := hg.1 hd
    have e := answer_alg c s cs hi hl h
    have hc : s'.coins = s.coins ++ cs := by rw [e]; simp [hon]
    have hd' : s'.done = s.done := by rw [e]
    exact ⟨good_congr hg hd' (by rw [e]) (by rw [e]) (fun _ => by rw [e]) (fun h' => by rw [hd] at h'; cases h')
      (by rw [e]), by rw [e], hd'.trans hd, fun _ => hc, (fun h' => by cases h')⟩
  · have fin : s'.done = s.done → s'.on = s.on → s'.l0 = s.l0 → s'.coins = s.coins → s'.prior = s.prior →
        Good c s' ∧ s'.prior = s.prior ∧ s'.done = true ∧ (true = false → s'.coins = s.coins ++ cs) ∧
          (true = true → s'.coins = s.coins) := fun a b l d e =>
      ⟨good_congr hg a b l (fun h' => by rw [hd] at h'; cases h') (fun _ => d) e, e, a.trans hd,
        (fun h' => by cases h'), fun _ => d⟩
    cases hi : s.inner with
    | some k =>
      obtain ⟨a, b, l, d, e⟩ := answer_proj (Or.inl (by rw [hi]; rfl)) h
      exact fin a b l d e
    | none =>
      cases hl : s.l0
      · have hon : s.on = false := by
          cases hon : s.on
          · rfl
          · have := (hg.2 hd).1 hon
            rw [hl] at this
            cases this
        have e := answer_alg c s cs hi hl h
        exact fin (by rw [e]) (by rw [e]) (by rw [e]) (by rw [e]; simp [hon]) (by rw [e])
      · obtain ⟨a, b, l, d, e⟩ := answer_proj (Or.inr hl) h
        exact fin a b l d e

theorem round_keep {c : Cfg} {s s' : Masker} {content out : ByteArray} {n : ℕ} (hg : Good c s)
    (h : _root_.Flock.Firewall.Frame.round c s content n = .ok (out, s')) : Keep c s s' ∧ s'.owed = some n ∧ 1 ≤ n := by
  unfold _root_.Flock.Firewall.Frame.round at h
  obtain ⟨⟨o, t, fr⟩, hr, h⟩ := bindV_ok h
  simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
  obtain ⟨-, rfl⟩ := h
  exact roundFree_keep hg hr

theorem good_stopped {c : Cfg} {s : Masker} (hg : Good c s) : Good c { s with stopped := true } :=
  good_of_same hg ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

open _root_.Flock.Firewall.Frame (Reply check) in
/-- **A round the frame takes or refuses keeps its phase and rank**, where the rep runs on: it owes the round's coins,
or, refused, none. -/
theorem step_round {c : Cfg} {st : _root_.Flock.Firewall.Frame.State} {n : ℕ} {r : ByteArray}
    {pl : _root_.Flock.Firewall.Frame.Place} (hg : Good c st.m) (ho : st.m.owed = none)
    (he : (_root_.Flock.Firewall.Frame.step c st (.round n r pl)).1.ends = false) :
    Keep c st.m (_root_.Flock.Firewall.Frame.step c st (.round n r pl)).2.m ∧
      ((_root_.Flock.Firewall.Frame.step c st (.round n r pl)).2.m.owed = none ∨
        ((_root_.Flock.Firewall.Frame.step c st (.round n r pl)).2.m.owed = some n ∧ 1 ≤ n)) := by
  obtain ⟨s, b⟩ := st
  dsimp only at hg ho he ⊢
  have hstop : Keep c s { s with stopped := true } := ⟨good_stopped hg, rfl, rfl, id⟩
  cases pl with
  | sched r' =>
    cases hc : check c s r r' n with
    | error e =>
      cases hs : e.startsWith _root_.Flock.Firewall.Frame.STOP <;>
        simp only [_root_.Flock.Firewall.Frame.step, ho, hc, hs, Reply.ends, Bool.false_eq_true, ↓reduceIte,
          Bool.true_eq_false] at he ⊢
      exact ⟨hstop, Or.inl trivial⟩
    | ok q =>
      obtain ⟨out, s', why⟩ := q
      simp only [_root_.Flock.Firewall.Frame.step, ho, hc]
      cases why with
      | none =>
        obtain ⟨k, o, p⟩ := round_keep hg (check_none hc)
        exact ⟨k, Or.inr ⟨o, p⟩⟩
      | some w =>
        obtain ⟨k, o, p⟩ := round_keep (good_stopped hg) (check_some hc)
        exact ⟨k, Or.inr ⟨o, p⟩⟩
  | fail =>
    simp only [_root_.Flock.Firewall.Frame.step, ho, bind, Except.bind] at he ⊢
    generalize hR : _root_.Flock.Firewall.Frame.round c _ r n = R at he ⊢
    cases R with
    | error e =>
      cases hs : e.startsWith _root_.Flock.Firewall.Frame.STOP <;>
        simp only [hs, Reply.ends, Bool.false_eq_true, ↓reduceIte, Bool.true_eq_false] at he ⊢
      exact ⟨⟨good_of_same hg ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩, rfl, rfl, id⟩, Or.inl trivial⟩
    | ok q =>
      obtain ⟨out, s'⟩ := q
      simp only [pure, Except.pure]
      obtain ⟨k, o, p⟩ := round_keep (s := { s with stopped := true, owed := none })
        (good_of_same hg ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩) hR
      exact ⟨k, Or.inr ⟨o, p⟩⟩
  | own =>
    simp only [_root_.Flock.Firewall.Frame.step, ho, bind, Except.bind] at he ⊢
    generalize hR : _root_.Flock.Firewall.Frame.round c _ r n = R at he ⊢
    cases R with
    | error e =>
      cases hs : e.startsWith _root_.Flock.Firewall.Frame.STOP <;>
        simp only [hs, Reply.ends, Bool.false_eq_true, ↓reduceIte, Bool.true_eq_false] at he ⊢
      exact ⟨⟨good_of_same hg ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩, rfl, rfl, id⟩, Or.inl trivial⟩
    | ok q =>
      obtain ⟨out, s'⟩ := q
      cases hn : s'.inner.isNone
      · simp only [hn, pure, Except.pure, Bool.false_eq_true, ↓reduceIte]
        obtain ⟨k, o, p⟩ := round_keep (good_of_same hg ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩) hR
        exact ⟨k, Or.inr ⟨o, p⟩⟩
      · simp only [hn, ↓reduceIte, throw, throwThe, MonadExceptOf.throw] at he ⊢
        split at he
        · simp [Reply.ends] at he
        · rename_i hs
          simp only [hs, Bool.false_eq_true, ↓reduceIte]
          exact ⟨⟨good_of_same hg ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩, rfl, rfl, id⟩, Or.inl trivial⟩

theorem ends_of_run {c : Cfg} {st : _root_.Flock.Firewall.Frame.State} {q : _root_.Flock.Firewall.Frame.Req}
    {L : List _root_.Flock.Firewall.Frame.Req} (h : (_root_.Flock.Firewall.Frame.run c st (q :: L)).2 = false) :
    (_root_.Flock.Firewall.Frame.step c st q).1.ends = false := by
  cases he : (_root_.Flock.Firewall.Frame.step c st q).1.ends
  · rfl
  · exfalso
    simp only [_root_.Flock.Firewall.Frame.run] at h
    rcases hs : _root_.Flock.Firewall.Frame.step c st q with ⟨a, st'⟩
    rw [hs] at he h
    simp only [he, ↓reduceIte, Bool.true_eq_false] at h

theorem run_cons_of {c : Cfg} {st : _root_.Flock.Firewall.Frame.State} {q : _root_.Flock.Firewall.Frame.Req}
    {L : List _root_.Flock.Firewall.Frame.Req} (h : (_root_.Flock.Firewall.Frame.step c st q).1.ends = false) :
    _root_.Flock.Firewall.Frame.run c st (q :: L) =
      _root_.Flock.Firewall.Frame.run c (_root_.Flock.Firewall.Frame.step c st q).2 L := by
  simp only [_root_.Flock.Firewall.Frame.run]
  rcases hs : _root_.Flock.Firewall.Frame.step c st q with ⟨a, st'⟩
  rw [hs] at h
  simp only [h, Bool.false_eq_true, ↓reduceIte]

/-! ## Along a rep's requests -/

/-- Rounds `0 … d − 1`'s coins `cs`, flattened. -/
def flatR (cs : ℕ → Array F128) (d : ℕ) : List F128 := (List.range d).flatMap fun i => (cs i).toList

theorem flatR_succ (cs : ℕ → Array F128) (d : ℕ) : flatR cs (d + 1) = flatR cs d ++ (cs d).toList := by
  simp [flatR, List.range_succ, List.flatMap_append]

/-- **A rep's requests after its first round**: round `k`'s coins `cs k`, then round `k + 1` (`q`), for `k < J`. -/
def pairs (cs : ℕ → Array F128) (q : ℕ → _root_.Flock.Firewall.Frame.Req) (J : ℕ) :
    List _root_.Flock.Firewall.Frame.Req :=
  (List.range J).flatMap fun k => [.coins (.ok (cs k)), q (k + 1)]

theorem pairs_succ (cs : ℕ → Array F128) (q : ℕ → _root_.Flock.Firewall.Frame.Req) (J : ℕ) :
    pairs cs q (J + 1) = pairs cs q J ++ [.coins (.ok (cs J)), q (J + 1)] := by
  simp [pairs, List.range_succ, List.flatMap_append]

theorem pairs_head (cs : ℕ → Array F128) (q : ℕ → _root_.Flock.Firewall.Frame.Req) :
    ∀ J, 0 < J → ∃ L, pairs cs q J = .coins (.ok (cs 0)) :: L
  | 0, h => absurd h (Nat.lt_irrefl 0)
  | J + 1, _ => by
    rw [pairs_succ]
    rcases Nat.eq_zero_or_pos J with rfl | hJ
    · exact ⟨[q 1], by simp [pairs]⟩
    · obtain ⟨L, hL⟩ := pairs_head cs q J hJ
      exact ⟨L ++ [.coins (.ok (cs J)), q (J + 1)], by rw [hL]; rfl⟩

/-- **The frame after round `k`**, round `i` squeezing `nn i` and answered `cs i`: its phase and rank (`Good`), the
earlier reps' points `p`, the coins answered so far each the round's count, the round's coins owed (or none, the round
refused), and the coins it holds those of the rounds up to the opening's label, all of them before it. -/
def Inv (c : Cfg) (p : Array (Array F128)) (cs : ℕ → Array F128) (nn : ℕ → ℕ) (k : ℕ) (s : Masker) : Prop :=
  Good c s ∧ s.prior = p ∧ (∀ i < k, (cs i).size = nn i ∧ 1 ≤ nn i) ∧
    (s.owed = none ∨ (s.owed = some (nn k) ∧ 1 ≤ nn k)) ∧
    ∃ d ≤ k, s.coins.toList = flatR cs d ∧ (s.done = false → d = k)

theorem pair_inv {c : Cfg} {p : Array (Array F128)} {cs : ℕ → Array F128} {nn : ℕ → ℕ}
    {q : ℕ → _root_.Flock.Firewall.Frame.Req} (hq : ∀ k, ∃ r pl, q k = .round (nn k) r pl) {k : ℕ}
    {S : _root_.Flock.Firewall.Frame.State} (hI : Inv c p cs nn k S.m)
    (h2 : (_root_.Flock.Firewall.Frame.run c S [.coins (.ok (cs k)), q (k + 1)]).2 = false) :
    Inv c p cs nn (k + 1) (_root_.Flock.Firewall.Frame.run c S [.coins (.ok (cs k)), q (k + 1)]).1.m := by
  obtain ⟨hg, hp, hsz, ho, d, hd, hc, hdk⟩ := hI
  obtain ⟨r, pl, hqk⟩ := hq (k + 1)
  rw [hqk] at h2 ⊢
  obtain ⟨s1, ha, he⟩ := run_coins_cons c S (cs k) _ h2
  rw [he] at h2 ⊢
  have hn : S.m.owed = some (nn k) ∧ 1 ≤ nn k := by
    rcases ho with ho | ho
    · exact absurd ha (answer_owed_none _ ho)
    · exact ho
  have hsize := answer_size hn.1 ha
  obtain ⟨g1, p1, d1, c1, c1'⟩ := answer_keep hg ha
  have o1 := answer_owed ha
  have hE := ends_of_run h2
  rw [run_cons_of hE]
  simp only [_root_.Flock.Firewall.Frame.run]
  obtain ⟨hk, ho'⟩ := step_round (st := { S with m := s1 }) g1 o1 hE
  generalize _root_.Flock.Firewall.Frame.step c { S with m := s1 } (.round (nn (k + 1)) r pl) = R at hk ho' ⊢
  obtain ⟨gk, ck, pk, dk⟩ := hk
  refine ⟨gk, pk.trans (p1.trans hp), fun i hi => ?_, ho', ?_⟩
  · rcases Nat.lt_succ_iff_lt_or_eq.1 hi with hi | rfl
    · exact hsz i hi
    · exact ⟨hsize, hn.2⟩
  · cases hdS : S.m.done
    · have hd0 := hdk hdS
      refine ⟨k + 1, le_refl _, ?_, fun _ => rfl⟩
      rw [ck, c1 hdS, Array.toList_append, hc, hd0, flatR_succ]
    · refine ⟨d, by omega, ?_, fun h' => ?_⟩
      · rw [ck, c1' hdS, hc]
      · have := dk (d1.trans hdS)
        rw [h'] at this
        cases this

theorem run_pairs {c : Cfg} {p : Array (Array F128)} {cs : ℕ → Array F128} {nn : ℕ → ℕ}
    {q : ℕ → _root_.Flock.Firewall.Frame.Req} (hq : ∀ k, ∃ r pl, q k = .round (nn k) r pl)
    (st : _root_.Flock.Firewall.Frame.State) (hI : Inv c p cs nn 0 st.m) :
    ∀ J, (_root_.Flock.Firewall.Frame.run c st (pairs cs q J)).2 = false →
      Inv c p cs nn J (_root_.Flock.Firewall.Frame.run c st (pairs cs q J)).1.m
  | 0, _ => by simpa [pairs, _root_.Flock.Firewall.Frame.run] using hI
  | J + 1, h2 => by
    rw [pairs_succ, run_append] at h2 ⊢
    cases hb : (_root_.Flock.Firewall.Frame.run c st (pairs cs q J)).2
    · simp only [hb, Bool.false_eq_true, ↓reduceIte] at h2 ⊢
      have hI' := run_pairs hq st hI J hb
      generalize (_root_.Flock.Firewall.Frame.run c st (pairs cs q J)).1 = S at h2 hI' ⊢
      exact pair_inv hq hI' h2
    · simp [hb] at h2

/-- **A rep's frame from its start along its requests**: from the earlier reps' points `p`, past the zerocheck's head
(round `0`, `roundFree_zcHead`) and then each round's coins and the next round, where the rep runs on, the frame keeps
its phase and rank (`Inv`). -/
theorem run_head {c : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) (p : Array (Array F128))
    {digest cap dom : ByteArray} (hdg : digest.size < 2 ^ 64) (hcap : cap.size < 2 ^ 64) (hdom : dom.size < 2 ^ 64)
    {cs : ℕ → Array F128} {nn : ℕ → ℕ} {q : ℕ → _root_.Flock.Firewall.Frame.Req}
    (hq : ∀ k, ∃ r pl, q k = .round (nn k) r pl) (h6 : nn 0 = 6) {J : ℕ} (hJ : 0 < J)
    (h2 : (_root_.Flock.Firewall.Frame.run c { m := { prior := p } } (.round 6 (ByteArray.mk (encs [.dom dom,
      .label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap, .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray)
      (.sched (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap,
        .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray)) :: pairs cs q J)).2 = false) :
    Inv c p cs nn J (_root_.Flock.Firewall.Frame.run c { m := { prior := p } } (.round 6 (ByteArray.mk (encs [.dom dom,
      .label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap, .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray)
      (.sched (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap,
        .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray)) :: pairs cs q J)).1.m := by
  rcases Loop.run_sched_cons (st := { m := { prior := p } }) rfl h2 with ⟨out, sx, why, hck, he⟩ | he
  · rw [he] at h2 ⊢
    have hsx : sx = { ({ prior := p } : Masker) with on := true, owed := some 6 } ∨
        sx = { ({ prior := p } : Masker) with stopped := true, on := true, owed := some 6 } := by
      rcases Loop.check_self hck with ⟨fr, hr⟩ | ⟨fr, hr⟩
      · obtain ⟨o, fr', hr'⟩ := roundFree_zcHead c { prior := p } digest cap dom hc rfl rfl rfl hdg hcap hdom
          (Nat.zero_le _)
        rw [hr'] at hr
        simp only [Except.ok.injEq, Prod.mk.injEq] at hr
        exact Or.inl hr.2.1.symm
      · obtain ⟨o, fr', hr'⟩ := roundFree_zcHead c { ({ prior := p } : Masker) with stopped := true } digest cap
          dom hc rfl rfl rfl hdg hcap hdom (Nat.zero_le _)
        rw [hr'] at hr
        simp only [Except.ok.injEq, Prod.mk.injEq] at hr
        exact Or.inr hr.2.1.symm
    refine run_pairs hq _ ?_ J h2
    rcases hsx with rfl | rfl <;>
      exact ⟨⟨fun _ => ⟨rfl, rfl, rfl⟩, fun h => by simp at h⟩, rfl, fun i hi => absurd hi (Nat.not_lt_zero _),
        Or.inr ⟨by rw [h6], by rw [h6]; decide⟩, 0, le_refl _, by simp [flatR], fun _ => rfl⟩
  · rw [he] at h2
    obtain ⟨L, hL⟩ := pairs_head cs q J hJ
    rw [hL] at h2
    obtain ⟨s1, ha, -⟩ := run_coins_cons c _ _ _ h2
    exact absurd ha (answer_owed_none _ rfl)

/-- Round `k`'s count in a rep's requests up to the inner proof (`ownReqs`): a scheduled round's, then the inner
proof's. -/
def nnOf (sched own : Array (ByteArray × ℕ)) (k : ℕ) : ℕ :=
  if k < sched.size then (sched.getD k (.empty, 0)).2 else (own.getD (k - sched.size) (.empty, 0)).2

/-- Round `k` of a rep's requests up to the inner proof (`ownReqs`): a scheduled round, with itself in the device's
place, then the inner proof's. -/
def qOf (sched own : Array (ByteArray × ℕ)) (k : ℕ) : _root_.Flock.Firewall.Frame.Req :=
  if k < sched.size then
    .round (sched.getD k (.empty, 0)).2 (sched.getD k (.empty, 0)).1 (.sched (sched.getD k (.empty, 0)).1)
  else .round (own.getD (k - sched.size) (.empty, 0)).2 (own.getD (k - sched.size) (.empty, 0)).1 .own

theorem qOf_round (sched own : Array (ByteArray × ℕ)) (k : ℕ) :
    ∃ r pl, qOf sched own k = .round (nnOf sched own k) r pl := by
  by_cases hk : k < sched.size
  · simp only [qOf, nnOf, hk, ↓reduceIte]
    exact ⟨_, _, rfl⟩
  · simp only [qOf, nnOf, hk, ↓reduceIte]
    exact ⟨_, _, rfl⟩

theorem schedReqs_pairs (P : Array F128) (sched own : Array (ByteArray × ℕ)) (cr : Array (Array F128)) :
    ∀ j, j < sched.size → schedReqs P sched cr j =
      (if P.isEmpty then [] else [.prior P.size (_root_.Flock.Firewall.Frame.wordBytes P)]) ++
        (qOf sched own 0 :: pairs (fun k => cr.getD k #[]) (qOf sched own) j)
  | 0, h => by simp [schedReqs, pairs, qOf, h]
  | j + 1, h => by
    rw [schedReqs_succ, schedReqs_pairs P sched own cr j (by omega), pairs_succ]
    simp [qOf, h]

/-- **A rep's requests up to the inner proof's round `i`** are its first round, then each round's coins and the next. -/
theorem ownReqs_pairs (P : Array F128) (sched own : Array (ByteArray × ℕ)) (cr : Array (Array F128))
    (hS : 0 < sched.size) : ∀ i, ownReqs P sched own cr i =
      (if P.isEmpty then [] else [.prior P.size (_root_.Flock.Firewall.Frame.wordBytes P)]) ++
        (qOf sched own 0 :: pairs (fun k => cr.getD k #[]) (qOf sched own) (sched.size + i))
  | 0 => by
    obtain ⟨j, hj⟩ : ∃ j, sched.size = j + 1 := ⟨sched.size - 1, by omega⟩
    rw [ownReqs_zero P sched own cr hj, schedReqs_pairs P sched own cr j (by omega), hj, Nat.add_zero, pairs_succ]
    simp [qOf, hj]
  | i + 1 => by
    rw [ownReqs_succ, ownReqs_pairs P sched own cr hS i, ← Nat.add_assoc, pairs_succ]
    have h1 : ¬ (sched.size + i + 1 < sched.size) := by omega
    have h2 : sched.size + i + 1 - sched.size = i + 1 := by omega
    simp [qOf, h1, h2]

/-- An earlier rep's coins, past which the rep runs on, are its claim points. -/
theorem run_prior_eq (c : Cfg) (P : Array F128) (rs : List _root_.Flock.Firewall.Frame.Req)
    (h2 : (_root_.Flock.Firewall.Frame.run c {} ((if P.isEmpty then [] else
      [.prior P.size (_root_.Flock.Firewall.Frame.wordBytes P)]) ++ rs)).2 = false) :
    ∃ p, (P.isEmpty = true → p = #[]) ∧
      (P.isEmpty = false → claimPoints c.lay
        (_root_.Flock.Firewall.Frame.words (_root_.Flock.Firewall.Frame.wordBytes P) 0 P.size) = .ok p) ∧
      _root_.Flock.Firewall.Frame.run c {} ((if P.isEmpty then [] else
        [.prior P.size (_root_.Flock.Firewall.Frame.wordBytes P)]) ++ rs) =
        _root_.Flock.Firewall.Frame.run c { m := { prior := p } } rs := by
  cases hP : P.isEmpty
  · simp only [hP, Bool.false_eq_true, ↓reduceIte, List.singleton_append] at h2 ⊢
    rcases hcp : claimPoints c.lay
        (_root_.Flock.Firewall.Frame.words (_root_.Flock.Firewall.Frame.wordBytes P) 0 P.size) with e | p
    · exfalso
      simp only [_root_.Flock.Firewall.Frame.run, _root_.Flock.Firewall.Frame.step, hcp,
        _root_.Flock.Firewall.Frame.Reply.ends] at h2
      cases h2
    · refine ⟨p, fun h => h.elim, fun _ => rfl, ?_⟩
      simp only [_root_.Flock.Firewall.Frame.run, _root_.Flock.Firewall.Frame.step, hcp,
        _root_.Flock.Firewall.Frame.Reply.ends, Bool.false_eq_true, ↓reduceIte]
      simp
  · exact ⟨#[], fun _ => rfl, (fun h => by cases h), by simp⟩

/-- **Past the opening's label the rank check held at the coins answered**: where a rep's requests up to the inner
proof's round `i` (`ownReqs`) run on, the rounds answered `cr`, into the inner proof, the frame took the opening's
label: it holds rounds `0 … d − 1`'s coins, each answered the round's count, and the rank check held there after the
earlier rep's claim points. -/
theorem ownReqs_rank {c : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) (P : Array F128)
    (sched own : Array (ByteArray × ℕ)) (cr : Array (Array F128)) {digest cap dom : ByteArray}
    (hdg : digest.size < 2 ^ 64) (hcap : cap.size < 2 ^ 64) (hdom : dom.size < 2 ^ 64)
    (h0 : sched.getD 0 (.empty, 0) = (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes digest,
      .bytes cap, .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray, 6)) (i : ℕ)
    (h2 : (_root_.Flock.Firewall.Frame.run c {} (ownReqs P sched own cr i)).2 = false)
    (hin : (_root_.Flock.Firewall.Frame.run c {} (ownReqs P sched own cr i)).1.m.inner.isSome = true) :
    ∃ p, (P.isEmpty = true → p = #[]) ∧
      (P.isEmpty = false → claimPoints c.lay
        (_root_.Flock.Firewall.Frame.words (_root_.Flock.Firewall.Frame.wordBytes P) 0 P.size) = .ok p) ∧
      RankHeld c (_root_.Flock.Firewall.Frame.run c {} (ownReqs P sched own cr i)).1.m ∧
      (_root_.Flock.Firewall.Frame.run c {} (ownReqs P sched own cr i)).1.m.prior = p ∧
      ∃ d ≤ sched.size + i, (_root_.Flock.Firewall.Frame.run c {} (ownReqs P sched own cr i)).1.m.coins.toList =
          flatR (fun k => cr.getD k #[]) d ∧
        ∀ k < sched.size + i, (cr.getD k #[]).size = nnOf sched own k ∧ 1 ≤ nnOf sched own k := by
  have hS : 0 < sched.size := by
    by_contra hz
    have e : sched.getD 0 (.empty, 0) = (.empty, 0) := by
      simp [Array.getD, show sched.size = 0 by omega]
    rw [e] at h0
    simp at h0
  rw [ownReqs_pairs P sched own cr hS i] at h2 hin ⊢
  obtain ⟨p, hp1, hp2, he⟩ := run_prior_eq c P _ h2
  rw [he] at h2 hin ⊢
  have hq0 : qOf sched own 0 = .round 6 (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8,
      .bytes digest, .bytes cap, .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray)
      (.sched (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap,
        .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray)) := by
    simp only [qOf, hS, ↓reduceIte, h0]
  rw [hq0] at h2 hin ⊢
  obtain ⟨hg, hpr, hsz, -, d, hd, hco, -⟩ := run_head hc p hdg hcap hdom (qOf_round sched own)
    (by simp only [nnOf, hS, ↓reduceIte, h0]) (J := sched.size + i) (by omega) h2
  have hdone : (_root_.Flock.Firewall.Frame.run c { m := { prior := p } } (.round 6 (ByteArray.mk (encs [.dom dom,
      .label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap, .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray)
      (.sched (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap,
        .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray)) ::
        pairs (fun k => cr.getD k #[]) (qOf sched own) (sched.size + i))).1.m.done = true := by
    cases hD : (_root_.Flock.Firewall.Frame.run c { m := { prior := p } } (.round 6 (ByteArray.mk (encs [.dom dom,
      .label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap, .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray)
      (.sched (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap,
        .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray)) ::
        pairs (fun k => cr.getD k #[]) (qOf sched own) (sched.size + i))).1.m.done
    · rw [(hg.1 hD).2.2] at hin
      cases hin
    · rfl
  exact ⟨p, hp1, hp2, (hg.2 hdone).2, hpr, d, hd, hco, hsz⟩

end Ranked

end Claims.ZKServer
