import Proofs.Flock.Claims.ZK.L0PinQuiet

/-!
# The frame's phase flags along a rep's scheduled rounds, up to its opening

`Ranked.Good` says what the frame's phase is at each state, but not where it is. Along a rep's scheduled rounds whose
rounds past the zerocheck's head are quiet (`Quiet.QRound`) but for one, the opening's first (`PcsRound`, the opening's
label and then quiet frames), the frame is in the algebraic phase before that round and, past it, outside every masked
phase at the algebraic phase's last position:

* `Rel`: two states alike in the phase flags, and in the position where the first is outside a masked phase. A plain
  label (`label_rel`), an absorb (`absorb_rel`) and so every quiet frame (`foldlM_stepAt_rel`) keep it, and so does a
  quiet round (`roundFree_quiet`), which owes its coins.
* `roundFree_pcs`: the opening's first round, in the algebraic phase, ends it, at the position the label demands
  (`c.nPiop`).
* `Fl`, `FInv`: the flags after round `k`, the opening's label at round `A`; or the frame owes nothing, the round
  refused.
* **`sched_flags`**, **`sched_opened`**: along `schedReqs` from the rep's start where the rep runs on, the frame keeps
  `FInv`; and where it takes the coins of a round at or past the opening's label, it is outside every masked phase, past
  the algebraic phase's opening, before level 0 and the inner proof, at the position `c.nPiop`, owing nothing.
-/

namespace Claims.ZKServer.Flags

open _root_.Flock (V F128)
open _root_.Flock.Firewall.Frame (Cfg Masker isLabel LABEL_PCS LABEL_ZC roundFree INNER_ROUNDS claimPoints check)
open FlockSoundness.Refine (Frame encs)
open Loop

/-- **Alike in the phase flags**: `s'` has `s`'s phase, level 0's and the inner proof's flags, and, where `s` is outside
a masked phase, its position. -/
def Rel (s s' : Masker) : Prop :=
  s'.done = s.done ∧ s'.on = s.on ∧ s'.l0 = s.l0 ∧ s'.doneL0 = s.doneL0 ∧ s'.inner = s.inner ∧
    (s.on = false → s'.pos = s.pos)

theorem rel_refl (s : Masker) : Rel s s := ⟨rfl, rfl, rfl, rfl, rfl, fun _ => rfl⟩

theorem rel_trans {s s' s'' : Masker} (h : Rel s s') (h' : Rel s' s'') : Rel s s'' := by
  obtain ⟨a1, a2, a3, a4, a5, a6⟩ := h
  obtain ⟨b1, b2, b3, b4, b5, b6⟩ := h'
  exact ⟨b1.trans a1, b2.trans a2, b3.trans a3, b4.trans a4, b5.trans a5, fun ho => (b6 (a2.trans ho)).trans (a6 ho)⟩

theorem absorb_rel {c : Cfg} {s : Masker} {v : F128} {q : Masker × F128} (h : s.absorb c v = .ok q) : Rel s q.1 := by
  unfold Masker.absorb at h
  cases hon : s.on
  · simp only [hon, Bool.not_false, ↓reduceIte, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact rel_refl s
  · simp only [hon, Bool.not_true, Bool.false_eq_true, ↓reduceIte] at h
    cases hi : c.order[s.pos]? with
    | none => simp only [hi] at h; cases h
    | some i =>
      cases hp : c.h[i]? with
      | none => simp only [hi, hp] at h; cases h
      | some p =>
        simp only [hi, hp, pure, Except.pure, Except.ok.injEq] at h
        subst h
        exact ⟨rfl, hon.symm, rfl, rfl, rfl, fun h' => by simp [hon] at h'⟩

theorem foldlM_absorb1_rel (c : Cfg) : ∀ (vs : List F128) (r r' : Masker × ByteArray × Array ℕ × ℕ),
    vs.foldlM (absorb1 c) r = .ok r' → Rel r.1 r'.1
  | [], r, r', h => by cases h; exact rel_refl _
  | v :: vs, r, r', h => by
    rw [List.foldlM_cons] at h
    cases ha : absorb1 c r v with
    | error e => rw [ha] at h; cases h
    | ok q =>
      rw [ha] at h
      have hq : Rel r.1 q.1 := by
        unfold absorb1 at ha
        cases hb : r.1.absorb c v with
        | error e => rw [hb] at ha; cases ha
        | ok b => rw [hb] at ha; cases ha; exact absorb_rel hb
      exact rel_trans hq (foldlM_absorb1_rel c vs q r' h)

theorem plainLabel_parts {b : ByteArray} (hp : Quiet.plainLabel b = true) :
    isLabel b LABEL_PCS = false ∧ isLabel b _root_.Flock.Zk.LABEL_CLOSE = false ∧
      isLabel b _root_.Flock.Zk.LABEL_INNER = false ∧ isLabel b _root_.Flock.Zk.LABEL_OPEN = false := by
  unfold Quiet.plainLabel at hp
  revert hp
  generalize isLabel b LABEL_PCS = a1
  generalize isLabel b _root_.Flock.Zk.LABEL_CLOSE = a2
  generalize isLabel b _root_.Flock.Zk.LABEL_INNER = a3
  generalize isLabel b _root_.Flock.Zk.LABEL_OPEN = a4
  cases a1 <;> cases a2 <;> cases a3 <;> cases a4 <;> decide

/-- **A plain label keeps the flags and the position**, in the algebraic phase or past it. -/
theorem label_rel {c : Cfg} {s : Masker} {b : ByteArray} {q : Masker × ByteArray} (hp : Quiet.plainLabel b = true)
    (hA : s.done = false → s.on = true) (h : s.label c b = .ok q) : Rel s q.1 := by
  obtain ⟨h1, h2, h3, h4⟩ := plainLabel_parts hp
  by_cases hZ : isLabel b LABEL_ZC = true
  · rw [Ranked.eq_of_isLabel hZ] at h
    obtain ⟨p1, p2, p3, p4⟩ := Ranked.zc_other
    cases hd : s.done
    · simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hd] at h
      cases h
      refine ⟨?_, ?_, ?_, ?_, ?_, fun _ => ?_⟩ <;> first | rfl | simp [hd, hA hd]
    · simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hd] at h
      cases h
      exact rel_refl s
  · rw [label_plain c s b ⟨h1, h2, h3, by simpa using hZ, h4⟩] at h
    cases h
    exact rel_refl s

/-- **Quiet frames keep the flags**: the frame's steps over plain frames from a state alike to `s₀`. -/
theorem foldlM_stepAt_rel (c : Cfg) (s₀ : Masker) (hA : s₀.done = false → s₀.on = true) :
    ∀ (fs : List Frame) (r r' : Masker × ByteArray × Array ℕ × ℕ), Rel s₀ r.1 → (∀ f ∈ fs, Quiet.Plain f) →
      fs.foldlM (stepAt c) r = .ok r' → Rel s₀ r'.1
  | [], r, r', hr, _, h => by cases h; exact hr
  | f :: fs, r, r', hr, hp, h => by
    rw [List.foldlM_cons] at h
    cases hs : stepAt c r f with
    | error e => rw [hs] at h; cases h
    | ok q =>
      rw [hs] at h
      have hA' : r.1.done = false → r.1.on = true := fun hd => hr.2.1.trans (hA (hr.1.symm.trans hd))
      have hf : Quiet.Plain f := hp f List.mem_cons_self
      have hq : Rel r.1 q.1 := by
        cases f with
        | dom b => cases hs; exact rel_refl _
        | label b =>
          simp only [stepAt] at hs
          cases hl : r.1.label c b with
          | error e => rw [hl] at hs; cases hs
          | ok q' =>
            rw [hl] at hs
            cases hs
            exact label_rel (hf.2 b rfl) hA' hl
        | bytes b => cases hs; exact rel_refl _
        | one v =>
          simp only [stepAt] at hs
          have hq' := foldlM_absorb1_rel c _ _ q hs
          exact hq'
        | vec vs =>
          simp only [stepAt] at hs
          have hq' := foldlM_absorb1_rel c _ _ q hs
          exact hq'
        | sq k n => cases hs; exact rel_refl _
      exact foldlM_stepAt_rel c s₀ hA fs q r' (rel_trans hr hq) (fun g hg => hp g (List.mem_cons_of_mem _ hg)) h

theorem eq_mk_encs {b : ByteArray} {l : List UInt8} (h : b.data.toList = l) : b = ByteArray.mk l.toArray := by
  rw [← h]

/-- **A quiet round keeps the flags** and owes its coins. -/
theorem roundFree_quiet {c : Cfg} {s : Masker} {r : ByteArray × ℕ} {q : ByteArray × Masker × Array ℕ}
    (hq : Quiet.QRound r) (hA : s.done = false → s.on = true) (hi : s.inner = none)
    (h : roundFree c s r.1 r.2 = .ok q) : Rel s q.2.1 ∧ q.2.1.owed = some r.2 := by
  obtain ⟨fs, k, hd, hp, hk, h0, hn⟩ := hq
  rw [eq_mk_encs hd, roundFree_encs c s fs k r.2 (by rw [hi]; simp) (fun f hf => (hp f hf).1) hk
    (fun h' => h0 h'.1) hn] at h
  obtain ⟨x, hx, h⟩ := bindV_ok h
  have hR := foldlM_stepAt_rel c s hA fs _ x (rel_refl s) hp hx
  have hi' : x.1.inner = none := hR.2.2.2.2.1.trans hi
  simp only [hi'] at h
  cases h
  refine ⟨⟨hR.1, hR.2.1, hR.2.2.1, hR.2.2.2.1, ?_, hR.2.2.2.2.2⟩, rfl⟩
  first | exact hi.symm | exact hR.2.2.2.2.1

/-- **The opening's label in the algebraic phase** demands its last position and ends it. -/
theorem label_pcs {c : Cfg} {s : Masker} {q : Masker × ByteArray} (hon : s.on = true)
    (h : s.label c LABEL_PCS.toUTF8 = .ok q) :
    s.pos = c.nPiop ∧ q.1.done = true ∧ q.1.on = false ∧ q.1.l0 = s.l0 ∧ q.1.doneL0 = s.doneL0 ∧
      q.1.inner = s.inner ∧ q.1.pos = s.pos := by
  obtain ⟨p1, p2, p3, p4⟩ := Ranked.pcs_other
  by_cases hpos : s.pos = c.nPiop
  · refine ⟨hpos, ?_⟩
    cases hm : c.mask with
    | none =>
      simp [Masker.label, hon, hm, Ranked.isLabel_self, p1, p2, p3, p4, hpos] at h
      cases h
      refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩ <;> first | rfl | simp [hpos]
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
          refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩ <;> first | rfl | simp [hpos]
        · cases h
  · cases hm : c.mask <;> simp [Masker.label, hon, hm, Ranked.isLabel_self, p1, p2, p3, p4, hpos] at h <;> cases h

/-- **The opening's first round**: its label, then quiet frames, and a squeeze of its count. -/
def PcsRound (r : ByteArray × ℕ) : Prop :=
  ∃ fs k, r.1.data.toList = encs (.label LABEL_PCS.toUTF8 :: fs) ++ (Frame.sq k r.2).enc ∧
    (∀ f ∈ fs, Quiet.Plain f) ∧ (k = 2 ∨ (k = 1 ∧ r.2 = 1)) ∧ r.2 ≠ 0 ∧ r.2 < 2 ^ 64

/-- **The opening's first round ends the algebraic phase** at its last position, `c.nPiop`, and owes its coins. -/
theorem roundFree_pcs {c : Cfg} {s : Masker} {r : ByteArray × ℕ} {q : ByteArray × Masker × Array ℕ}
    (hq : PcsRound r) (hon : s.on = true) (hi : s.inner = none) (h : roundFree c s r.1 r.2 = .ok q) :
    s.pos = c.nPiop ∧ q.2.1.done = true ∧ q.2.1.on = false ∧ q.2.1.l0 = s.l0 ∧ q.2.1.doneL0 = s.doneL0 ∧
      q.2.1.inner = none ∧ q.2.1.pos = s.pos ∧ q.2.1.owed = some r.2 := by
  obtain ⟨fs, k, hd, hp, hk, h0, hn⟩ := hq
  have hm : ∀ f ∈ Frame.label LABEL_PCS.toUTF8 :: fs, midFrame f := by
    intro f hf
    rcases List.mem_cons.1 hf with rfl | hf
    · show LABEL_PCS.toUTF8.size < 2 ^ 64
      decide
    · exact (hp f hf).1
  rw [eq_mk_encs hd, roundFree_encs c s _ k r.2 (by rw [hi]; simp) hm hk (fun h' => h0 h'.1) hn] at h
  obtain ⟨x, hx, h⟩ := bindV_ok h
  rw [List.foldlM_cons] at hx
  obtain ⟨y, hy, hx⟩ := bindV_ok hx
  obtain ⟨q', hl, hyq⟩ : ∃ q', s.label c LABEL_PCS.toUTF8 = .ok q' ∧ y.1 = q'.1 := by
    simp only [stepAt] at hy
    cases hl : s.label c LABEL_PCS.toUTF8 with
    | error e => rw [hl] at hy; cases hy
    | ok q' =>
      rw [hl] at hy
      cases hy
      exact ⟨q', rfl, rfl⟩
  obtain ⟨hpos, hdn, hof, hl0, hdl, hin, hps⟩ := label_pcs hon hl
  rw [← hyq] at hdn hof hl0 hdl hin hps
  have hR := foldlM_stepAt_rel c y.1 (fun h' => by rw [hdn] at h'; cases h') fs y x (rel_refl _) hp hx
  have hi' : x.1.inner = none := hR.2.2.2.2.1.trans (hin.trans hi)
  simp only [hi'] at h
  cases h
  refine ⟨hpos, hR.1.trans hdn, hR.2.1.trans hof, hR.2.2.1.trans hl0, hR.2.2.2.1.trans hdl, ?_,
    (hR.2.2.2.2.2 hof).trans hps, rfl⟩
  first | rfl | exact hi'

/-! ## Along a rep's scheduled rounds -/

/-- **The frame's flags after round `k`**, the opening's label at round `A`: outside level 0 and the inner proof, in the
algebraic phase before `A`, and past it outside every masked phase at the algebraic phase's last position. -/
def Fl (c : Cfg) (A k : ℕ) (s : Masker) : Prop :=
  s.l0 = false ∧ s.doneL0 = false ∧ s.inner = none ∧ (k < A → s.done = false ∧ s.on = true) ∧
    (A ≤ k → s.done = true ∧ s.on = false ∧ s.pos = c.nPiop)

/-- The flags after round `k`, or the frame owes nothing (the round refused), where the rep can run on only to its
end. -/
def FInv (c : Cfg) (A k : ℕ) (s : Masker) : Prop := s.owed = none ∨ (s.owed.isSome = true ∧ Fl c A k s)

theorem fl_congr {c : Cfg} {A k : ℕ} {s s' : Masker} (hF : Fl c A k s) (h1 : s'.l0 = s.l0)
    (h2 : s'.doneL0 = s.doneL0) (h3 : s'.inner = s.inner) (h4 : s'.done = s.done) (h5 : s'.on = s.on)
    (h6 : s'.pos = s.pos) : Fl c A k s' := by
  obtain ⟨a1, a2, a3, a4, a5⟩ := hF
  refine ⟨h1.trans a1, h2.trans a2, h3.trans a3, fun hk => ?_, fun hk => ?_⟩
  · obtain ⟨b1, b2⟩ := a4 hk
    exact ⟨h4.trans b1, h5.trans b2⟩
  · obtain ⟨b1, b2, b3⟩ := a5 hk
    exact ⟨h4.trans b1, h5.trans b2, h6.trans b3⟩

/-- **Round `k + 1` keeps the flags**: quiet, or the opening's first at `A`. -/
theorem flags_round {c : Cfg} {A k : ℕ} {s : Masker} {r : ByteArray × ℕ} {q : ByteArray × Masker × Array ℕ}
    (hF : Fl c A k s) (hq : k + 1 ≠ A → Quiet.QRound r) (hp : k + 1 = A → PcsRound r)
    (h : roundFree c s r.1 r.2 = .ok q) : Fl c A (k + 1) q.2.1 ∧ q.2.1.owed = some r.2 := by
  obtain ⟨a1, a2, a3, a4, a5⟩ := hF
  by_cases hA : k + 1 = A
  · obtain ⟨b1, b2⟩ := a4 (by omega)
    obtain ⟨hpos, hdn, hof, hl0, hdl, hin, hps, ho⟩ := roundFree_pcs (hp hA) b2 a3 h
    exact ⟨⟨hl0.trans a1, hdl.trans a2, hin, fun hk => absurd hk (by omega), fun _ => ⟨hdn, hof, hps.trans hpos⟩⟩,
      ho⟩
  · have hAl : s.done = false → s.on = true := by
      intro hd
      by_cases hk : k < A
      · exact (a4 hk).2
      · rw [(a5 (by omega)).1] at hd
        cases hd
    obtain ⟨⟨r1, r2, r3, r4, r5, r6⟩, ho⟩ := roundFree_quiet (hq hA) hAl a3 h
    refine ⟨⟨r3.trans a1, r4.trans a2, r5.trans a3, fun hk => ?_, fun hk => ?_⟩, ho⟩
    · obtain ⟨b1, b2⟩ := a4 (by omega)
      exact ⟨r1.trans b1, r2.trans b2⟩
    · obtain ⟨b1, b2, b3⟩ := a5 (by omega)
      exact ⟨r1.trans b1, r2.trans b2, (r6 b2).trans b3⟩

/-- **The coins keep the flags**, outside level 0 and the inner proof, and leave the frame owing nothing. -/
theorem flags_answer {c : Cfg} {A k : ℕ} {s s' : Masker} {cs : Array F128} (hF : Fl c A k s)
    (h : s.answer c cs = .ok s') : Fl c A k s' ∧ s'.owed = none := by
  have e := answer_alg c s cs hF.2.2.1 hF.1 h
  subst e
  exact ⟨fl_congr hF rfl rfl rfl rfl rfl rfl, rfl⟩

/-- **A scheduled round after the coins**: from flags after round `k` and owing round `k`'s coins, the coins and round
`k + 1` in its own place keep `FInv`, where the rep runs on. -/
theorem flags_pair {c : Cfg} {A k : ℕ} {S : _root_.Flock.Firewall.Frame.State} {cs : Array F128}
    {r : ByteArray × ℕ} (hI : FInv c A k S.m) (hq : k + 1 ≠ A → Quiet.QRound r) (hp : k + 1 = A → PcsRound r)
    (h2 : (_root_.Flock.Firewall.Frame.run c S [.coins (.ok cs), .round r.2 r.1 (.sched r.1)]).2 = false) :
    FInv c A (k + 1) (_root_.Flock.Firewall.Frame.run c S [.coins (.ok cs), .round r.2 r.1 (.sched r.1)]).1.m := by
  obtain ⟨s1, ha, he⟩ := run_coins_cons c S cs _ h2
  rw [he] at h2 ⊢
  have hF : Fl c A k S.m := by
    rcases hI with ho | ⟨-, hF⟩
    · exact absurd ha (answer_owed_none _ ho)
    · exact hF
  obtain ⟨hF1, ho1⟩ := flags_answer hF ha
  rcases Loop.run_sched_cons (st := { S with m := s1 }) ho1 h2 with ⟨out, sx, why, hck, he'⟩ | he'
  · rw [he']
    show FInv c A (k + 1) sx
    rcases Loop.check_self hck with ⟨fr, hr⟩ | ⟨fr, hr⟩
    · obtain ⟨hF', ho'⟩ := flags_round (s := s1) (q := (out, sx, fr)) hF1 hq hp hr
      exact Or.inr ⟨by rw [show sx.owed = some r.2 from ho']; rfl, hF'⟩
    · obtain ⟨hF', ho'⟩ := flags_round (s := { s1 with stopped := true }) (q := (out, sx, fr))
        (fl_congr hF1 rfl rfl rfl rfl rfl rfl) hq hp hr
      exact Or.inr ⟨by rw [show sx.owed = some r.2 from ho']; rfl, hF'⟩
  · rw [he']
    exact Or.inl ho1

theorem schedReqs_zero (P : Array F128) (sched : Array (ByteArray × ℕ)) (cr : Array (Array F128)) :
    schedReqs P sched cr 0 =
      (if P.isEmpty then [] else [.prior P.size (_root_.Flock.Firewall.Frame.wordBytes P)]) ++
        [.round (sched.getD 0 (.empty, 0)).2 (sched.getD 0 (.empty, 0)).1 (.sched (sched.getD 0 (.empty, 0)).1)] := by
  unfold schedReqs
  rw [List.range_zero, List.flatMap_nil, List.append_nil]

/-- **A rep's frame along its scheduled rounds keeps its flags**: from the rep's start, past the zerocheck's head (round
`0`) and rounds `1 … J`, quiet but for the opening's first at `A`, each after the coins of the one before, where the rep
runs on, the frame keeps `FInv`. -/
theorem sched_flags {c : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) (P : Array F128)
    (sched : Array (ByteArray × ℕ)) (cr : Array (Array F128)) {digest cap dom : ByteArray}
    (hdg : digest.size < 2 ^ 64) (hcap : cap.size < 2 ^ 64) (hdom : dom.size < 2 ^ 64)
    (h0 : sched.getD 0 (.empty, 0) = (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes digest,
      .bytes cap, .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray, 6))
    {A : ℕ} (hA0 : 0 < A) (hp : PcsRound (sched.getD A (.empty, 0))) :
    ∀ J, (∀ k, 1 ≤ k → k ≤ J → k ≠ A → Quiet.QRound (sched.getD k (.empty, 0))) →
      (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr J)).2 = false →
      FInv c A J (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr J)).1.m
  | 0, _, h2 => by
    rw [schedReqs_zero, h0] at h2 ⊢
    dsimp only at h2 ⊢
    obtain ⟨p, -, -, he⟩ := Ranked.run_prior_eq c P _ h2
    rw [he] at h2 ⊢
    rcases Loop.run_sched_cons (st := { m := { prior := p } }) rfl h2 with ⟨out, sx, why, hck, he'⟩ | he'
    · rw [he']
      show FInv c A 0 sx
      have hsx : ∃ b, sx = { ({ prior := p } : Masker) with stopped := b, on := true, owed := some 6 } := by
        rcases Loop.check_self hck with ⟨fr, hr⟩ | ⟨fr, hr⟩
        · obtain ⟨o, fr', hr'⟩ := roundFree_zcHead c { prior := p } digest cap dom hc rfl rfl rfl hdg hcap hdom
            (Nat.zero_le _)
          rw [hr'] at hr
          simp only [Except.ok.injEq, Prod.mk.injEq] at hr
          exact ⟨false, hr.2.1.symm⟩
        · obtain ⟨o, fr', hr'⟩ := roundFree_zcHead c { ({ prior := p } : Masker) with stopped := true } digest cap
            dom hc rfl rfl rfl hdg hcap hdom (Nat.zero_le _)
          rw [hr'] at hr
          simp only [Except.ok.injEq, Prod.mk.injEq] at hr
          exact ⟨true, hr.2.1.symm⟩
      obtain ⟨b, rfl⟩ := hsx
      exact Or.inr ⟨rfl, rfl, rfl, rfl, fun _ => ⟨rfl, rfl⟩, fun h => absurd h (by omega)⟩
    · rw [he']
      exact Or.inl rfl
  | J + 1, hq, h2 => by
    rw [schedReqs_succ, Loop.run_append] at h2 ⊢
    cases hb : (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr J)).2
    · simp only [hb, Bool.false_eq_true, ↓reduceIte] at h2 ⊢
      have hI := sched_flags hc P sched cr hdg hcap hdom h0 hA0 hp J
        (fun k h1 hk hA => hq k h1 (by omega) hA) hb
      exact flags_pair (r := sched.getD (J + 1) (.empty, 0)) hI (hq (J + 1) (by omega) le_rfl)
        (fun hA => by subst hA; exact hp) h2
    · simp [hb] at h2

/-- **Past the opening's first round, at the coins**: where a rep's scheduled rounds up to `J`, at or past the opening's
first round `A`, and then round `J`'s coins run on, the frame is outside every masked phase, past the algebraic phase's
opening, before level 0 and the inner proof, at the algebraic phase's last position `c.nPiop`, and owes nothing. -/
theorem sched_opened {c : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) (P : Array F128)
    (sched : Array (ByteArray × ℕ)) (cr : Array (Array F128)) {digest cap dom : ByteArray}
    (hdg : digest.size < 2 ^ 64) (hcap : cap.size < 2 ^ 64) (hdom : dom.size < 2 ^ 64)
    (h0 : sched.getD 0 (.empty, 0) = (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes digest,
      .bytes cap, .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray, 6))
    {A J : ℕ} (hA0 : 0 < A) (hAJ : A ≤ J) (hp : PcsRound (sched.getD A (.empty, 0)))
    (hq : ∀ k, 1 ≤ k → k ≤ J → k ≠ A → Quiet.QRound (sched.getD k (.empty, 0)))
    (h2 : (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr J ++ [.coins (.ok (cr.getD J #[]))])).2 =
      false) :
    let m := (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr J ++ [.coins (.ok (cr.getD J #[]))])).1.m
    m.done = true ∧ m.on = false ∧ m.l0 = false ∧ m.doneL0 = false ∧ m.inner = none ∧ m.pos = c.nPiop ∧
      m.owed = none := by
  intro m
  rw [Loop.run_append] at h2
  cases hb : (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr J)).2
  · simp only [hb, Bool.false_eq_true, ↓reduceIte] at h2
    have hI := sched_flags hc P sched cr hdg hcap hdom h0 hA0 hp J hq hb
    have hm : m = (_root_.Flock.Firewall.Frame.run c (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr J)).1
        [.coins (.ok (cr.getD J #[]))]).1.m := by
      show (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr J ++ [.coins (.ok (cr.getD J #[]))])).1.m = _
      rw [Loop.run_append, hb]
      rfl
    rw [hm]
    generalize (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr J)).1 = S at h2 hI ⊢
    obtain ⟨s1, ha, he⟩ := run_coins_cons c S _ [] h2
    rw [he]
    show s1.done = true ∧ s1.on = false ∧ s1.l0 = false ∧ s1.doneL0 = false ∧ s1.inner = none ∧
      s1.pos = c.nPiop ∧ s1.owed = none
    have hF : Fl c A J S.m := by
      rcases hI with ho | ⟨-, hF⟩
      · exact absurd ha (answer_owed_none _ ho)
      · exact hF
    obtain ⟨⟨a1, a2, a3, -, a5⟩, ho⟩ := flags_answer hF ha
    obtain ⟨b1, b2, b3⟩ := a5 hAJ
    exact ⟨b1, b2, a1, a2, a3, b3, ho⟩
  · simp [hb] at h2

end Claims.ZKServer.Flags
