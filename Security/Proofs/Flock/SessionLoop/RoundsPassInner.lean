module

public import Proofs.Flock.SessionLoop.RoundsPassTaken
public import Proofs.Flock.SessionLoop.RoundsPassFrame
public import Proofs.Flock.SessionLoop.FinHonest
public import Proofs.Flock.SessionLoop.LiveInner

/-!
# The inner proof's rounds 1 to 3 pass at `valsDev`

Past a rep's schedule `withInner`'s session loop sends the frame's own inner round (`Frame.innerSched`) at the state
`innerOf` builds from the stream's coins: its `ε` the coins after the inner proof's round 0, its `β` those after its
round 2. At its rounds 1 to 3 the frame checks that round at the state the stream's coins leave, which holds the same
`ε` and `β`, outside every masked phase, so it frees no word; `valsDev` sends the round as it came (round 2's `τ`
commitment being the honest one `finHonest` holds), and the frame's check passes it.

* `Loop.absorb_epsBeta`, `Loop.label_epsBeta`, `Loop.roundFree_keeps`, `Loop.roundFree_epsBeta`: an absorbed value, a
  label and a round keep the frame's `ε` and `β`; `Loop.answer_epsBeta`: inside the inner proof the coins after its
  round 0 are `ε`, after its round 2 `β`.
* `Loop.run_ownReqs_epsBeta`, `Loop.repState_own_epsBeta`: along a rep's requests into the inner proof, and at the
  state the coins after its round `i` leave, `ε` and `β` are the coins `innerOf` reads.
* `Loop.ownRound_state_congr`, `Loop.innerSched_state_congr`: the frame's own round reads the state only at round 1's
  `ε` and round 3's `β`. `Loop.roundFree_ownAt`, `Loop.ownRound_frames`, `Loop.ownRound_head`: outside every masked
  phase the frame takes its own round 1 to 3 freeing no word, the round is frames and a squeeze, and none is round 0.
* `Loop.setFinalC_of_head`, `Loop.setTau_of_ne`, `Loop.setTau_inner2_self`, `Loop.valsDev_own`: `valsDev` sends such a
  round as it came. `Loop.passes_self`, `Loop.roundOk_valsDev_own`: the frame's check passes it.
  `Loop.roundOk_ownAt`: so the round passes the frame's checks at `valsDev` wherever `cross` passes it.
* `LeanRelease.ofLoop_answered_epsBeta`: at the session loop's own frames, past a rep's schedule the state the frame
  takes the coins at holds `innerOf`'s `ε`, and from round 3 on its `β`. `LeanRelease.roundOk_innerAt`: at the inner
  proof's rounds 1 to 3 of `withInner`'s session loop on `finHonest`, the round passes the frame's checks at
  `valsDev honestVals`.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)
open _root_.Flock.Transcript (header)
open _root_.Flock.Firewall.Frame (Cfg Masker isLabel roundFree INNER_ROUNDS check labelFrame wordBytes words innerSched
  ownRound passes firstDiff LABEL_PCS LABEL_ZC claimPoints)
open _root_.Flock.Firewall.Finish (Tab)
open FlockSoundness.Refine (Frame encs)

namespace Loop

/-! ## Rounds keep `ε` and `β` -/

/-- An absorbed value keeps the frame's `ε` and `β`. -/
theorem absorb_epsBeta {c : Cfg} {s : Masker} {v : F128} {q : Masker × F128} (h : s.absorb c v = .ok q) :
    (q.1.eps, q.1.beta) = (s.eps, s.beta) := by
  unfold Masker.absorb at h
  cases hon : s.on
  · simp only [hon, Bool.not_false, ↓reduceIte, pure, Except.pure, Except.ok.injEq] at h
    subst h
    rfl
  · simp only [hon, Bool.not_true, Bool.false_eq_true, ↓reduceIte] at h
    cases hi : c.order[s.pos]? with
    | none => simp only [hi] at h; cases h
    | some i =>
      cases hp : c.h[i]? with
      | none => simp only [hi, hp] at h; cases h
      | some p =>
        simp only [hi, hp, pure, Except.pure, Except.ok.injEq] at h
        subst h
        rfl

/-- A label keeps the frame's `ε` and `β`. -/
theorem label_epsBeta {c : Cfg} {s : Masker} {l : ByteArray} {q : Masker × ByteArray} (h : s.label c l = .ok q) :
    (q.1.eps, q.1.beta) = (s.eps, s.beta) := by
  by_cases hP : isLabel l LABEL_PCS = true
  · rw [Ranked.eq_of_isLabel hP] at h
    cases hon : s.on
    · have e : s.label c LABEL_PCS.toUTF8 = .ok (s, _root_.Flock.Firewall.Frame.labelFrame LABEL_PCS.toUTF8) := by
        obtain ⟨p1, p2, p3, p4⟩ := Ranked.pcs_other
        simp [Masker.label, hon, Ranked.isLabel_self, p1, p2, p3, p4]
        rfl
      rw [e] at h
      cases h
      rfl
    · obtain ⟨p1, p2, p3, p4⟩ := Ranked.pcs_other
      by_cases hpos : s.pos = c.nPiop
      · cases hm : c.mask with
        | none =>
          simp [Masker.label, hon, hm, Ranked.isLabel_self, p1, p2, p3, p4, hpos] at h
          cases h
          rfl
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
              rfl
            · cases h
      · cases hm : c.mask <;> simp [Masker.label, hon, hm, Ranked.isLabel_self, p1, p2, p3, p4, hpos] at h <;>
          cases h
  by_cases hC : isLabel l _root_.Flock.Zk.LABEL_CLOSE = true
  · rw [Ranked.eq_of_isLabel hC] at h
    obtain ⟨p1, p2, p3, p4⟩ := Ranked.close_other
    cases hl0 : s.l0
    · simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hl0] at h
      cases h
      rfl
    · by_cases hpos : s.pos = c.order.size
      · by_cases hrho : s.rho.size = _root_.Flock.Firewall.Frame.RHO_COINS
        · simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hl0, hpos, hrho] at h
          cases h
          rfl
        · simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hl0, hpos, hrho] at h
          cases h
      · simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hl0, hpos] at h
        cases h
  by_cases hI : isLabel l _root_.Flock.Zk.LABEL_INNER = true
  · rw [Ranked.eq_of_isLabel hI] at h
    obtain ⟨p1, p2, p3, p4⟩ := Ranked.inner_other
    cases hi : s.inner with
    | some k =>
      simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hi] at h
      cases h
    | none =>
      cases hd : s.done
      · simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hi, hd] at h
        cases h
      · cases hon : s.on
        · simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hi, hd, hon] at h
          split at h
          · cases h
          · cases h
            rfl
        · simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hi, hd, hon] at h
          cases h
  by_cases hZ : isLabel l LABEL_ZC = true
  · rw [Ranked.eq_of_isLabel hZ] at h
    obtain ⟨p1, p2, p3, p4⟩ := Ranked.zc_other
    cases hd : s.done
    · simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hd] at h
      cases h
      rfl
    · simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hd] at h
      cases h
      rfl
  by_cases hO : isLabel l _root_.Flock.Zk.LABEL_OPEN = true
  · rw [Ranked.eq_of_isLabel hO] at h
    obtain ⟨p1, p2, p3, p4⟩ := Ranked.open_other
    simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4] at h
    split at h
    · cases h
      rfl
    · cases h
      rfl
  simp only [Bool.not_eq_true] at hP hC hI hZ hO
  rw [label_plain c s l ⟨hP, hC, hI, hZ, hO⟩] at h
  cases h
  rfl

/-- **A round keeps what its labels, absorbed values and squeeze keep**: where every label and absorbed value keeps
`f` of the frame, and so does owing a round's coins, so does a round, whatever its bytes. -/
theorem roundFree_keeps {α : Type} (f : Masker → α) {c : Cfg}
    (hlab : ∀ {s : Masker} {l : ByteArray} {q : Masker × ByteArray}, s.label c l = .ok q → f q.1 = f s)
    (habs : ∀ {s : Masker} {v : F128} {q : Masker × F128}, s.absorb c v = .ok q → f q.1 = f s)
    (hown : ∀ (s : Masker) (n : ℕ), f { s with owed := some n } = f s) {s : Masker} {content : ByteArray} {n : ℕ}
    {q : ByteArray × Masker × Array ℕ} (h : roundFree c s content n = .ok q) : f q.2.1 = f s := by
  unfold roundFree at h
  by_cases hI : (s.inner == some INNER_ROUNDS) = true
  · simp only [hI, ↓reduceIte] at h
    cases h
  simp only [hI, Bool.false_eq_true, ↓reduceIte, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at h
  obtain ⟨r, hr, h⟩ := bindV_ok h
  have hQ := Ranked.forIn_inv (fun x : Option (ByteArray × Masker × Array ℕ) × Masker × ByteArray × Array ℕ × ℕ =>
      f x.2.1 = f s ∧ ∀ v, x.1 = some v → f v.2.1 = f s) _ ?step _ _ _
    ⟨rfl, fun _ hv => by cases hv⟩ hr
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
    · have hy := Ranked.ite_bind_ok hy
      cases hy
      exact ⟨hk, fun _ hv => by cases hv⟩
    · have hy := Ranked.ite_bind_ok hy
      obtain ⟨x, hx, hy⟩ := bindV_ok hy
      cases hy
      exact ⟨(hlab hx).trans hk, fun _ hv => by cases hv⟩
    · have hy := Ranked.ite_bind_ok hy
      generalize (_root_.Flock.u64OfLE content (pos + 8)).toNat = len at hy
      generalize content.get! (pos + 1) = kind at hy
      by_cases h3 : (!(kind == 2 || kind == 1 && len == 1)) = true
      · rw [ite_eq_left h3] at hy
        cases hy
      rw [ite_eq_right h3] at hy
      obtain ⟨r', hr', hy⟩ := bindV_ok hy
      cases hy
      have hQ' : f r'.1 = f s := by
        refine Ranked.forIn_inv (fun x : Masker × ByteArray × Array ℕ => f x.1 = f s) _ ?_ _ _ _ hk hr'
        intro i x y hx hy
        by_cases h8 : (x.1.on && decide (x.1.pos < c.nPiop)) = true
        · rw [ite_eq_left h8] at hy
          obtain ⟨a, ha, hy⟩ := bindV_ok hy
          cases hy
          exact (habs ha).trans hx
        · rw [ite_eq_right h8] at hy
          obtain ⟨a, ha, hy⟩ := bindV_ok hy
          cases hy
          exact (habs ha).trans hx
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
      split at hy
      · obtain ⟨u, -, hy⟩ := bindV_ok hy
        cases hy
        exact ⟨hk, fun v hv => by cases hv; exact (hown _ _).trans hk⟩
      · cases hy
        exact ⟨hk, fun v hv => by cases hv; exact (hown _ _).trans hk⟩
    · have hy := Ranked.ite_bind_ok hy
      cases hy
      exact ⟨hk, fun _ hv => by cases hv⟩
    · cases hy

/-- A round keeps the frame's `ε` and `β`, whatever its bytes. -/
theorem roundFree_epsBeta {c : Cfg} {s : Masker} {content : ByteArray} {n : ℕ} {q : ByteArray × Masker × Array ℕ}
    (h : roundFree c s content n = .ok q) : (q.2.1.eps, q.2.1.beta) = (s.eps, s.beta) :=
  roundFree_keeps (fun s => (s.eps, s.beta)) label_epsBeta absorb_epsBeta (fun _ _ => rfl) h

/-- **Inside the inner proof the coins after its round 0 are `ε`, after its round 2 `β`**: the frame's answer at its
round `k` sets `ε` at `k = 0`, `β` at `k = 2`, and keeps both otherwise. -/
theorem answer_epsBeta {c : Cfg} {s s' : Masker} {cs : Array F128} {k : ℕ} (hs : s.inner = some k)
    (h : s.answer c cs = .ok s') :
    s'.eps = (if k = 0 then cs[0]! else s.eps) ∧ s'.beta = (if k = 2 then cs[0]! else s.beta) := by
  unfold Masker.answer at h
  simp only [bind, Except.bind, pure, Except.pure, hs] at h
  repeat' split at h
  all_goals first
    | (cases h; done)
    | (cases h; by_cases h0 : k = 0 <;> by_cases h2 : k = 2 <;>
        simp_all [_root_.Flock.Firewall.Frame.INNER_BETA])

/-! ## `ε` and `β` along a rep's requests -/

/-- **`ε` and `β` along the loop's requests into the inner proof**: where the requests up to the inner proof's round
`i` neither end the rep nor leave it owing no coins, from round 1 on `ε` is the coins after round 0, and from round 3 on
`β` the coins after round 2. -/
theorem run_ownReqs_epsBeta (c : Cfg) (prior : Array F128) (sched own : Array (ByteArray × ℕ))
    (cr : Array (Array F128))
    (h0 : ∃ s w t n, w.size = 16 ∧ t.size = 64 ∧ own.getD 0 (.empty, 0) = (ownRound c s w t n 0, n)) :
    ∀ i, (_root_.Flock.Firewall.Frame.run c {} (ownReqs prior sched own cr i)).2 = false →
      (_root_.Flock.Firewall.Frame.run c {} (ownReqs prior sched own cr i)).1.m.owed ≠ none →
      (1 ≤ i → (_root_.Flock.Firewall.Frame.run c {} (ownReqs prior sched own cr i)).1.m.eps =
        (cr.getD sched.size #[])[0]!) ∧
      (3 ≤ i → (_root_.Flock.Firewall.Frame.run c {} (ownReqs prior sched own cr i)).1.m.beta =
        (cr.getD (sched.size + 2) #[])[0]!)
  | 0, _, _ => ⟨fun h => absurd h (by omega), fun h => absurd h (by omega)⟩
  | i + 1, h2, ho => by
    rw [ownReqs_succ] at h2 ho ⊢
    obtain ⟨hb, s1, ha, out, s', fr, hr, -, hm⟩ := run_snoc_coins_own c {} _ _ _ _ h2 ho
    have hI := run_ownReqs_inner c prior sched own cr h0 i hb (fun e => answer_owed_none _ e ha)
    have hIH := run_ownReqs_epsBeta c prior sched own cr h0 i hb (fun e => answer_owed_none _ e ha)
    obtain ⟨he, hbt⟩ := answer_epsBeta hI ha
    have hk := roundFree_epsBeta hr
    rw [hm]
    refine ⟨fun _ => ?_, fun h3 => ?_⟩
    · rw [show s'.eps = s1.eps from congrArg Prod.fst hk, he]
      split_ifs with h
      · subst h
        rfl
      · exact hIH.1 (by omega)
    · rw [show s'.beta = s1.beta from congrArg Prod.snd hk, hbt]
      split_ifs with h
      · subst h
        rfl
      · exact hIH.2 (by omega)

/-- **`ε` and `β` at the loop's state inside the inner proof**: where the frame takes the coins after the inner proof's
round `i` at the rep's state, its `ε` is the coins after round 0, and from round 2 on its `β` the coins after round 2. -/
theorem repState_own_epsBeta (c c' : Cfg) (t : Tab) (rep : ℕ) (dom : ByteArray) (xs : Array (Array F128))
    (hi : Array ByteArray) (cr0 cr : Array (Array F128)) (ns : ℕ → ℕ) {i : ℕ} {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr.pop = .ok sched) (hi4 : i < INNER_ROUNDS) {cs : Array F128} {m : Masker}
    (hcs : cs = cr.getD (sched.size + i) #[])
    (ha : (repState c t rep dom xs hi cr0 cr ns (sched.size + i + 1)).answer c' cs = .ok m) :
    m.eps = (cr.getD sched.size #[])[0]! ∧ (2 ≤ i → m.beta = (cr.getD (sched.size + 2) #[])[0]!) := by
  rw [repState_own_at c t rep dom xs hi cr0 cr ns i hal hs hi4] at ha
  dsimp only at ha
  generalize hR : _root_.Flock.Firewall.Frame.run c {}
    (ownReqs (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched
      (ownRounds c (cr.extract sched.size cr.size) fun j => ns (sched.size + j)) cr i) = R at ha
  obtain ⟨S, b⟩ := R
  cases b with
  | true =>
    simp only [↓reduceIte] at ha
    exact absurd ha (answer_owed_none cs rfl)
  | false =>
    simp only [Bool.false_eq_true, ↓reduceIte] at ha
    have hI := run_ownReqs_inner c _ sched _ cr ⟨_, zeros 16, zeros 64, _, zeros_size 16, zeros_size 64,
      ownRounds_getD c _ _ (show 0 < INNER_ROUNDS by decide)⟩ i (by rw [hR])
      (by rw [hR]; exact fun e => answer_owed_none cs e ha)
    have hE := run_ownReqs_epsBeta c _ sched _ cr ⟨_, zeros 16, zeros 64, _, zeros_size 16, zeros_size 64,
      ownRounds_getD c _ _ (show 0 < INNER_ROUNDS by decide)⟩ i (by rw [hR])
      (by rw [hR]; exact fun e => answer_owed_none cs e ha)
    rw [hR] at hI hE
    obtain ⟨he, hb⟩ := answer_epsBeta hI ha
    subst hcs
    refine ⟨?_, fun h2 => ?_⟩
    · rw [he]
      split_ifs with h
      · subst h
        rfl
      · exact hE.1 (by omega)
    · rw [hb]
      split_ifs with h
      · subst h
        rfl
      · exact hE.2 (by omega)

/-! ## The frame's own rounds 1 to 3 -/

/-- **The frame's own round reads the state only at round 1's `ε` and round 3's `β`.** -/
theorem ownRound_state_congr (c : Cfg) {S s : Masker} (w t : ByteArray) (n : ℕ) {k : ℕ} (hk : k < 4)
    (h1 : k = 1 → S.eps = s.eps) (h3 : k = 3 → S.beta = s.beta) : ownRound c S w t n k = ownRound c s w t n k := by
  obtain rfl | rfl | rfl | rfl : k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 := by omega
  · simp only [ownRound]
  · simp only [ownRound, h1 rfl]
  · simp only [ownRound]
  · simp only [ownRound, h3 rfl]

theorem innerSched_state_congr (c : Cfg) {S s : Masker} (fin : Array ByteArray) (n : ℕ) {k : ℕ} (hk : k < 4)
    (h1 : k = 1 → S.eps = s.eps) (h3 : k = 3 → S.beta = s.beta) :
    innerSched c S fin n k = innerSched c s fin n k := by
  unfold innerSched
  exact ownRound_state_congr c _ _ n hk h1 h3

theorem ownSq_at {n k : ℕ} (hk0 : 0 < k) (hk4 : k < 4) (hn0 : n ≠ 0) (hn1 : k = 2 → n = 1) :
    ∃ k' : UInt8, ownSq n k = Frame.sq k' n ∧ (k' = 2 ∨ (k' = 1 ∧ n = 1)) ∧ ¬ (n = 0 ∧ k' = 2) := by
  obtain rfl | rfl | rfl : k = 1 ∨ k = 2 ∨ k = 3 := by omega
  · exact ⟨2, rfl, .inl rfl, fun h => hn0 h.1⟩
  · obtain rfl := hn1 rfl
    exact ⟨1, rfl, .inr ⟨rfl, rfl⟩, fun h => hn0 h.1⟩
  · exact ⟨2, rfl, .inl rfl, fun h => hn0 h.1⟩

/-- **Outside every masked phase the frame takes its own round 1 to 3 freeing no word**: at the inner proof's round
`k`, at a coin count the round may squeeze, the round leaves with the state owing its coins. -/
theorem roundFree_ownAt {c : Cfg} {s : Masker} {w t : ByteArray} (hw : w.size = 16) (ht : t.size = 64) {k n : ℕ}
    (hk0 : 0 < k) (hk4 : k < 4) (hon : s.on = false) (hin : s.inner = some k) (hrow : c.rowH.size < 2 ^ 64)
    (hn : n < 2 ^ 64) (hn0 : n ≠ 0) (hn1 : k = 2 → n = 1) :
    ∃ o, roundFree c s (ownRound c s w t n k) n = .ok (o, { s with owed := some n }, #[]) := by
  obtain ⟨k', hkm, hk', h0⟩ := ownSq_at hk0 hk4 hn0 hn1
  have hs : s.inner ≠ some INNER_ROUNDS := by
    rw [hin]
    intro e
    have e' : k = 4 := Option.some.inj e
    omega
  have hX := ownRound_encs (c := c) (s := s) (n := n) hw ht hk4
  rw [hkm] at hX
  have e : ownRound c s w t n k = ByteArray.mk (encs (ownFrames c s w t k) ++ (Frame.sq k' n).enc).toArray := by
    rw [← hX, Array.toArray_toList]
  obtain ⟨r, hr, h1, -⟩ := foldlM_stepAt_off c (ownFrames c s w t k) s ByteArray.empty #[] 0 0 hon
    (ownFrames_plain c s w t hk0)
  have hfr := foldlM_stepAt_off_fr c (ownFrames c s w t k) s ByteArray.empty #[] 0 r hon
    (ownFrames_plain c s w t hk0) hr
  refine ⟨r.2.1 ++ header 4 k' n, ?_⟩
  rw [e, roundFree_encs c s _ k' n hs (ownFrames_mid hrow s w t ht hk0) hk' h0 hn, hr]
  simp only [bind, Except.bind, h1, hin, hfr]
  rw [← e, innerOwn_ownRound c s w t hw ht n k hk4]
  try rfl

theorem ownRound_frames {c : Cfg} {s : Masker} {w t : ByteArray} (hw : w.size = 16) (ht : t.size = 64) {k n : ℕ}
    (hk0 : 0 < k) (hk4 : k < 4) (hrow : c.rowH.size < 2 ^ 64) (hn : n < 2 ^ 64) (hn0 : n ≠ 0)
    (hn1 : k = 2 → n = 1) : Frames (ownRound c s w t n k).data.toList n := by
  obtain ⟨k', hkm, hk', h0⟩ := ownSq_at hk0 hk4 hn0 hn1
  refine ⟨ownFrames c s w t k, k', ?_, ownFrames_mid hrow s w t ht hk0, hk', h0, hn⟩
  rw [ownRound_encs hw ht hk4, hkm]

/-- The frame's own rounds 1 to 3 start with no label frame, so none is round 0's shape. -/
theorem ownRound_head (c : Cfg) (s : Masker) (w t : ByteArray) (n : ℕ) {k : ℕ} (hk0 : 0 < k) (hk4 : k < 4) :
    (ownRound c s w t n k).data.toList.head? ≠ some 2 := by
  obtain rfl | rfl | rfl : k = 1 ∨ k = 2 ∨ k = 3 := by omega
  all_goals simp only [ownRound, ByteArray.toList_data_append, List.append_assoc, head_header_append]
  all_goals decide

/-! ## `valsDev` at the frame's own rounds -/

theorem setFinalC_of_head {u : List UInt8} (h : u.head? ≠ some 2) (a : F128) : setFinalC a u = u := by
  unfold setFinalC
  split_ifs with hc
  · obtain ⟨w', -, rfl⟩ := hc
    exact absurd (inner0_head w') h
  · rfl

theorem setTau_of_ne {m : Masker} (h : m.inner ≠ some 2) (t : ByteArray) (u : List UInt8) : setTau m t u = u := by
  unfold setTau
  split_ifs with hc
  · exact absurd hc.1 h
  · rfl

theorem setTau_inner2_self (m : Masker) (t : ByteArray) :
    setTau m t (inner2 t).data.toList = (inner2 t).data.toList := by
  unfold setTau
  split_ifs <;> rfl

/-- A round the frame takes passes the frame's check against itself. -/
theorem passes_self {c : Cfg} {s : Masker} {v : ByteArray} {n : ℕ} {o : ByteArray} {t : Masker} {fr : Array ℕ}
    (h : roundFree c s v n = .ok (o, t, fr)) : passes c s v v n = .ok none := by
  have hfd : firstDiff v v fr = none := HonestPass.firstDiff_eq_none fun _ _ _ => rfl
  unfold passes
  rw [h]
  simp only [bind, Except.bind, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte, hfd, Option.map_none]
  try rfl

/-- **`valsDev` sends a round that frees no word as it came**: where the frame took the stream's coins at `m` and
takes the scheduled round `r` there freeing no word, `r` frames and a squeeze, neither the `final_c'` `valsDev` writes
nor its `τ` changing it, `valsDev`'s round is `r`. -/
theorem valsDev_own (vs : finS.SwVals) (L : LeanRelease Flock.Wit Tp) {x : Msg} {w : Flock.Wit} {ρ : Tp}
    {g : List (Msg × Msg)} {m : Masker} {c : Cfg} {n : ℕ} {r o : ByteArray} {t : Masker}
    (hm : L.answered x ρ g = .ok m) (hc : L.cfg x ρ g = c) (hn : L.coins x g.length = n)
    (hsc : L.scheduled x w ρ g (computedOf (L.computed x w ρ g)) = r.data.toList)
    (hq : roundFree c { m with stopped := false } r n = .ok (o, t, #[])) (hF : Frames r.data.toList n)
    (h0 : setFinalC (vsAt vs L x w ρ g c.lay.fc + c.h.getD c.lay.fc 0) r.data.toList = r.data.toList)
    (h2 : setTau m (vsTauAt vs L x w ρ g) r.data.toList = r.data.toList) :
    valsDev vs L x w ρ g (computedOf (L.computed x w ρ g)) = r.data.toList := by
  have hr : (⟨r.data.toList.toArray⟩ : ByteArray) = r := by rw [Array.toArray_toList]
  have hfill : ∀ f, fill r.data.toList #[] f = r.data.toList := fun _ => rfl
  unfold valsDev
  rw [hm]
  simp only
  rw [hsc, hr, hc, hn, hq]
  simp only
  rw [ite_eq_left hF, hfill, h0, h2]

/-- **A round `valsDev` sends as it came passes the frame's checks** wherever `cross` passes it: where the frame took
the stream's coins at `m` and takes the scheduled round `r` there, a rep not stopped, and `valsDev` sends `r`. -/
theorem roundOk_valsDev_own (vs : finS.SwVals) (L : LeanRelease Flock.Wit Tp) {x : Msg} {w : Flock.Wit} {ρ : Tp}
    {g : List (Msg × Msg)} {m : Masker} {c : Cfg} {n : ℕ} {r o : ByteArray} {t : Masker} {fr : Array ℕ}
    (hm : L.answered x ρ g = .ok m) (hc : L.cfg x ρ g = c) (hn : L.coins x g.length = n)
    (hsc : L.scheduled x w ρ g (computedOf (L.computed x w ρ g)) = r.data.toList)
    (hd : valsDev vs L x w ρ g (computedOf (L.computed x w ρ g)) = r.data.toList)
    (hq : roundFree c { m with stopped := false } r n = .ok (o, t, fr))
    (hcr : L.cross x g.length w ρ g (valsDev vs L x w ρ g (computedOf (L.computed x w ρ g))) = true) :
    (L.withDevice (valsDev vs L)).roundOk x w ρ g = true := by
  have hr : (⟨r.data.toList.toArray⟩ : ByteArray) = r := by rw [Array.toArray_toList]
  have hc' := HonestPass.check_of_passes (s := { m with stopped := false }) rfl (passes_self hq)
    (round_of_roundFree hq)
  have hp : L.pass x w ρ g (valsDev vs L x w ρ g (computedOf (L.computed x w ρ g))) = true := by
    rw [hd]
    unfold LeanRelease.pass
    rw [hm]
    simp only
    rw [hsc, hr, hc, hn, hc']
  have ht : L.takes x ρ g (valsDev vs L x w ρ g (computedOf (L.computed x w ρ g))) = true := by
    rw [hd]
    unfold LeanRelease.takes LeanRelease.step
    rw [hm]
    simp only [bind, Except.bind]
    rw [hr, hc, hn, round_of_roundFree hq]
  rw [LeanRelease.roundOk_withDevice, Bool.and_eq_true, Bool.and_eq_true]
  exact ⟨⟨hcr, hp⟩, ht⟩

theorem extract_getD {α : Type} (c : Array α) (d : α) {s e a : ℕ} (h : s + a < e) (he : e ≤ c.size) :
    (c.extract s e).getD a d = c.getD (s + a) d := by
  have h1 : a < (c.extract s e).size := by rw [Array.size_extract]; omega
  have h2 : s + a < c.size := by omega
  rw [Array.getD_eq_getD_getElem?, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem h1,
    Array.getElem?_eq_getElem h2, Array.getElem_extract]

/-- **The frame's own inner round 1 to 3 passes the frame's checks at `valsDev`** wherever `cross` passes it: where
the frame took the stream's coins at `m`, inside the inner proof at round `k` outside every masked phase, and the
scheduled round is the frame's own round `k` at `m` (round 2's `τ` written by `valsDev` being the one it holds), at a
coin count the round may squeeze. -/
theorem roundOk_ownAt (vs : finS.SwVals) (L : LeanRelease Flock.Wit Tp) {x : Msg} {w : Flock.Wit} {ρ : Tp}
    {g : List (Msg × Msg)} {m : Masker} {c : Cfg} {fin : Array ByteArray} {n k : ℕ}
    (hm : L.answered x ρ g = .ok m) (hc : L.cfg x ρ g = c) (hn : L.coins x g.length = n)
    (hsc : L.scheduled x w ρ g (computedOf (L.computed x w ρ g)) =
      (innerSched c { m with stopped := false } fin n k).data.toList)
    (hk0 : 0 < k) (hk4 : k < 4) (hon : m.on = false) (hin : m.inner = some k) (hrow : c.rowH.size < 2 ^ 64)
    (hn64 : n < 2 ^ 64) (hn0 : n ≠ 0) (hn1 : k = 2 → n = 1)
    (h2 : k = 2 → setTau m (vsTauAt vs L x w ρ g) (innerSched c { m with stopped := false } fin n 2).data.toList =
      (innerSched c { m with stopped := false } fin n 2).data.toList)
    (hcr : L.cross x g.length w ρ g (valsDev vs L x w ρ g (computedOf (L.computed x w ρ g))) = true) :
    (L.withDevice (valsDev vs L)).roundOk x w ρ g = true := by
  have hw : (wordBytes (words (fin.getD 0 .empty)
      ((labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8).size + 16) 1)).size = 16 := by
    simp only [wordBytes_size, Loop.words_size]
  have ht : (wordBytes (words (fin.getD 2 .empty) 16 4)).size = 64 := by
    simp only [wordBytes_size, Loop.words_size]
  obtain ⟨o, hq⟩ := roundFree_ownAt (c := c) (s := { m with stopped := false }) hw ht hk0 hk4 hon hin hrow hn64 hn0
    hn1
  have hF : Frames (innerSched c { m with stopped := false } fin n k).data.toList n :=
    ownRound_frames hw ht hk0 hk4 hrow hn64 hn0 hn1
  have h0 : ∀ a, setFinalC a (innerSched c { m with stopped := false } fin n k).data.toList =
      (innerSched c { m with stopped := false } fin n k).data.toList :=
    setFinalC_of_head (ownRound_head _ _ _ _ _ hk0 hk4)
  have h2' : setTau m (vsTauAt vs L x w ρ g) (innerSched c { m with stopped := false } fin n k).data.toList =
      (innerSched c { m with stopped := false } fin n k).data.toList := by
    by_cases he : k = 2
    · subst he
      exact h2 rfl
    · exact setTau_of_ne (by rw [hin]; intro e; exact he (Option.some.inj e)) _ _
  have hd := valsDev_own vs L hm hc hn hsc hq hF (h0 _) h2'
  exact roundOk_valsDev_own vs L hm hc hn hsc hd hq hcr

end Loop

namespace LeanRelease

open Loop in
set_option maxHeartbeats 1000000 in
/-- **At the loop's own frames, past a rep's schedule the frame takes coins at `innerOf`'s `ε` and `β`**: where the
frame takes the coins after `e0 :: h1` at `m` and the next round is rep stream `k`'s round `j + 1`, past the end of its
schedule so far on its own table and salts, at the inner proof's round 1 to 3, `m`'s `ε` is the coins after the inner
proof's round 0, and from its round 3 on `m`'s `β` the coins after its round 2 (`repState_own_epsBeta`), the words
`innerOf` builds its state from. -/
theorem ofLoop_answered_epsBeta (T : CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (x : Msg) (w : Flock.Wit) (hSv : Serves x)
    (hR : ∀ (ρ : (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand)
      (s0 : _) (spec : SessionSpec) (sts : Array Stmt),
      FlockDeploy.setupOf true x = some (s0, spec, sts) → Count.RealTabs sts (real x w ρ))
    (ρ : (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) {m : Masker}
    (hA : (ofLoop T L₀ real).answered x ρ (e0 :: h1) = .ok m) (hne : (e0 :: h1).length ≠ lastOf x)
    {cs : List FlockDeploy.Coin} (hcs : h1.map (fun p => CoinLayer.rd p.2) = cs)
    {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q k j : ℕ}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h1.length = q + 1)
    (hki : CoinLayer.locate (roundsOf spec sts) 0 q = (k, j + 1))
    {tr : Tab} {hr : Array ByteArray} (hreal : real x w ρ k = .ok (tr, hr)) {rsched : Array (ByteArray × ℕ)}
    (hrs0 : schedSoFar tr (k % 2) (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8
      (_root_.Flock.Firewall.Drive.extras sts[k / 2]! (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]!
        (wordsAt x cs 0))) hr ((Array.range (j + 1)).map fun i => wordsAt x cs (q - (j + 1) + i + 1)) = .ok rsched)
    (hlt : rsched.size < j + 1) (h3 : j - rsched.size < 3) :
    m.eps = ((((Array.range (j + 1)).map fun i => wordsAt x cs (q - (j + 1) + i + 1)).extract rsched.size
        (j + 1)).getD 0 #[]).getD 0 0 ∧
      (2 ≤ j - rsched.size → m.beta = ((((Array.range (j + 1)).map fun i => wordsAt x cs (q - (j + 1) + i + 1)).extract
        rsched.size (j + 1)).getD 2 #[]).getD 0 0) := by
  have hq1 : cs.length = q + 1 := by rw [← hcs, List.length_map, hq]
  have hq2 : (e0 :: h1).length = q + 2 := by simp only [List.length_cons, hq]
  have hk1 : (CoinLayer.locate (roundsOf spec sts) 0 q).1 = k := by rw [hki]
  have hj : (CoinLayer.locate (roundsOf spec sts) 0 q).2 = j + 1 := by rw [hki]
  obtain ⟨m0, r0⟩ := e0
  have hcr : coinsAt x ((m0, r0) :: h1) = readCoins x cs := by
    rw [← hcs]
    exact loop_coinsRead loopNr T (withFrames T L₀ real) x m0 r0 h1
  have hF : (ofLoop T L₀ real).frameAnswer x ρ ((m0, r0) :: h1) = .ok m := by
    unfold answered at hA
    split at hA
    · exact hA
    · cases hA
  rw [ofLoop_frameAnswer_same T L₀ real x ρ (m0, r0) h1 hx hq hj, hcs, hcr, readCoins_eq x cs hq1] at hF
  replace hF := (bind_rankStop_ok hF).1
  obtain ⟨q', rfl⟩ : ∃ q', q = q' + 1 := by
    rcases q with _ | q'
    · have := locate_zero (roundsOf spec sts) 0
      rw [hki] at this
      exact absurd this (Nat.succ_ne_zero _)
    · exact ⟨q', rfl⟩
  have hp := locate_pred (roundsOf spec sts) 0 q' j hj
  obtain ⟨c', t, dom, xs, hi, hph⟩ :=
    progOf_placeholder x cs hx (p := q') (by rw [hq1]) (fun ho => answer_owed_none _ ho hF)
  rw [hp, hki] at hph
  have hph' : placeholder spec sts k (wordsAt x cs 0) = .ok (c', t, dom, xs, hi) := hph
  rw [progOf_at x cs hx hq1 hj (by rw [hk1]; exact hph'), hj, hk1] at hF
  have hqs := loop_round_in_plan T (withFrames T L₀ real) x ρ ((m0, r0) :: h1) hA hne hx hq2
  have hk : k < spec.repStreams.size := by
    have := locate_lt (roundsOf spec sts) 0 (q' + 1) hqs
    rw [hki] at this
    simpa [roundsOf] using this
  obtain ⟨_, _, _, _, hdm, hxe⟩ := placeholder_ok hph'
  obtain ⟨-, hd, hxr, -⟩ := sizesOk_at (sizesOk_of_serves hSv s0 spec sts hx) hk
  have hls : (wordsAt x cs 0).size = min (loopNr x 0) _root_.Flock.Firewall.Frames.COIN_MAX := by
    rw [← hcs]
    cases h1 with
    | nil => simp at hq
    | cons p h2 => simp [wordsAt, FlockDeploy.Coin.vals]
  have hdom : dom.size < 2 ^ 64 := by rw [hdm]; exact hd
  have hxs : xs.size + 2 < 2 ^ 64 := by
    rw [hxe, extras_size]
    have := Nat.mul_le_mul_left sts[k / 2]!.regions.size
      ((linkPoints_size_le sts[k / 2]! (wordsAt x cs 0)).trans (hls.trans_le (Nat.min_le_right _ _)))
    omega
  have hT := Count.real_tabAlike (hR ρ s0 spec sts hx) hph' hreal
  obtain ⟨hdg, hmb, hn, hlq, hpd, hdd⟩ := Count.real_bounds hph' hT
  have hR' := Sched.schedSoFar_alike tr t (k % 2) dom xs hr hi
    ((Array.range (j + 1)).map fun i => wordsAt x cs (q' + 1 - (j + 1) + i + 1))
    hT.1 hT.2.1 hT.2.2.1 hT.2.2.2.1 hT.2.2.2.2 hdom hdg hmb hxs hn hlq hpd hdd
  rw [← hdm, ← hxe] at hrs0
  rw [hrs0] at hR'
  cases hs' : schedSoFar t (k % 2) dom xs hi
      ((Array.range (j + 1)).map fun i => wordsAt x cs (q' + 1 - (j + 1) + i + 1)) with
  | error e => rw [hs'] at hR'; exact (hR' : False).elim
  | ok sched' =>
  rw [hs'] at hR'
  have hsz := roundsAt_size hR'
  obtain ⟨alg, hal⟩ := schedSoFar_ok_rounds hs'
  obtain ⟨sched, hs⟩ := schedSoFar_pop_ok hs'
  have hss : sched' = sched := schedSoFar_pop_eq (j := j) hs hs' (by simp) (by omega)
  rw [hss] at hsz
  have hjq : j + 1 ≤ q' + 1 := by
    have := locate_snd_le (roundsOf spec sts) 0 (q' + 1)
    rw [hki] at this
    exact this
  obtain ⟨i, rfl⟩ : ∃ i, j = sched.size + i := ⟨j - sched.size, by omega⟩
  have h4 : i < INNER_ROUNDS := by
    show i < 4
    omega
  have hlast : wordsAt x cs (q' + 1) = ((Array.range (sched.size + i + 1)).map fun i' =>
      wordsAt x cs (q' + 1 - (sched.size + i + 1) + i' + 1)).getD (sched.size + i) #[] := by
    simp only [Array.getD_eq_getD_getElem?, Array.getElem?_map, Array.getElem?_range,
      show sched.size + i < sched.size + i + 1 by omega, ↓reduceIte, Option.map_some, Option.getD_some]
    congr 1
    omega
  obtain ⟨he, hb⟩ := Loop.repState_own_epsBeta c' _ t (k % 2) dom xs hi _ _ _ hal hs h4 hlast hF
  have hsz' : rsched.size = sched.size := by omega
  have hcz : sched.size + i + 1 ≤ ((Array.range (sched.size + i + 1)).map fun i' =>
      wordsAt x cs (q' + 1 - (sched.size + i + 1) + i' + 1)).size := by
    rw [Array.size_map, Array.size_range]
  refine ⟨?_, fun h2 => ?_⟩
  · rw [hsz', Loop.extract_getD (s := sched.size) (a := 0) _ _ (by omega) hcz, Nat.add_zero,
      ← FlockSoundness.Refine.getElem!_eq_getD]
    exact he
  · rw [hsz', Loop.extract_getD (s := sched.size) (a := 2) _ _ (by omega) hcz,
      ← FlockSoundness.Refine.getElem!_eq_getD]
    exact hb (by omega)

open Loop in
set_option maxHeartbeats 1000000 in
/-- **At the inner proof's rounds 1 to 3 `withInner`'s session loop on `finHonest` passes the frame's checks at
`valsDev honestVals`**, wherever `cross` passes its device round: at a history at the inner proof's rounds
(`Loop.InnerAt`'s components) past the inner proof's round 0, on the rep's own tables where `real` answers the loop's,
at a `Live` statement the loop serves, where the frame takes the history's coins. The round in the device's place is
the frame's own inner round at `innerOf`'s state, which is the frame's own round at the state the coins leave
(`ofLoop_answered_epsBeta`), past the inner proof's label, outside every masked phase (`ofLoop_answered_on`,
`ofLoop_answered_inner`), at the coin counts the loop asks the inner proof's rounds for (`Loop.innerAt_slot`); at
round 2 `valsDev` writes the honest `τ` commitment, which is `finHonest`'s (`Loop.innerSched_finHonest2`). -/
theorem roundOk_innerAt (L₁ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Tp → ℕ → V (Tab × Array ByteArray))
    (fin : Msg → Flock.Wit → Tp → List (Msg × Msg) → Array ByteArray)
    (hfin : fin = finHonest (ofLoopLink L₁ src real)) {x : Msg} {w : Flock.Wit} (hx : Masked.Live x)
    (hSv : Serves x)
    (hR : ∀ (ρ : Tp) (s0 : _) (spec : SessionSpec) (sts : Array Stmt),
      FlockDeploy.setupOf true x = some (s0, spec, sts) → Count.RealTabs sts (real x w ρ))
    (ν : finS.Nonce) (t : finS.Tape) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hlt : ops.length < refK x)
    (href : (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) =
      false)
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} {q k i : ℕ} {tr : Tab} {hr : Array ByteArray}
    {rsched : Array (ByteArray × ℕ)} (hxs : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hq : (ops.map fun p => CoinLayer.rd p.2).length = q + 1)
    (hloc : CoinLayer.locate (roundsOf spec sts) 0 q = (k, i)) (hreal : real x w (ν, t) k = .ok (tr, hr))
    (hrs : schedSoFar tr (k % 2) (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8
      (_root_.Flock.Firewall.Drive.extras sts[k / 2]!
        (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]! (wordsAt x (ops.map fun p => CoinLayer.rd p.2) 0)))
      hr ((Array.range i).map fun j => wordsAt x (ops.map fun p => CoinLayer.rd p.2) (q - i + j + 1)) = .ok rsched)
    (hpos : rsched.size < i)
    (hcr : (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).cross x ((m₀, ρ₀) :: ops).length w (ν, t)
      ((m₀, ρ₀) :: ops) (valsDev finS.honestVals (ofLoopLink (withInner Masked.dataE L₁ real fin) src real) x w
        (ν, t) ((m₀, ρ₀) :: ops) (computedOf ((ofLoopLink (withInner Masked.dataE L₁ real fin) src real).computed x w
          (ν, t) ((m₀, ρ₀) :: ops)))) = true) :
    ((ofLoopLink (withInner Masked.dataE L₁ real fin) src real).withDevice
      (valsDev finS.honestVals (ofLoopLink (withInner Masked.dataE L₁ real fin) src real))).roundOk x w (ν, t)
      ((m₀, ρ₀) :: ops) = true := by
  subst hfin
  have hsch : schedOf (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2) = none :=
    schedOf_none_of_innerAt ⟨s0, spec, sts, q, k, i, tr, hr, rsched, hxs, hq, hloc, hreal, hrs, hpos.le⟩
  have hq1 : ops.length = q + 1 := by rw [← hq, List.length_map]
  obtain ⟨jf, rf, hkjr, he4, hslot, -, -, -, -, hnr⟩ := Loop.innerAt_slot hx hSv hxs hq
    (by rw [List.length_map]; exact hlt) hloc (hR (ν, t) s0 spec sts hxs) hreal hrs hpos.le
  have hin := Loop.innerOf_eq (real x w (ν, t)) ((withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops))
    (finHonest (ofLoopLink L₁ src real) x w (ν, t) ((m₀, ρ₀) :: ops)) x _ hxs hq hloc hreal hrs hpos.le he4
  have hsc := scheduled_innerAt L₁ src real (finHonest (ofLoopLink L₁ src real)) x w (ν, t) (m₀, ρ₀) ops hsch hin
  obtain ⟨m, ha⟩ := answered_of_unrefused href
  have hne : ((m₀, ρ₀) :: ops).length ≠ lastOf x := by rw [lastOf_eq, List.length_cons]; omega
  obtain ⟨j, rfl⟩ : ∃ j, i = j + 1 := ⟨i - 1, by omega⟩
  obtain ⟨-, hin', hrow⟩ := ofLoop_answered_inner Masked.dataE
    (withFiles (withLink (withInner Masked.dataE L₁ real (finHonest (ofLoopLink L₁ src real))) src) src) real x w hSv
    hR (ν, t) (m₀, ρ₀) ops ha hne rfl hxs hq1 hloc hreal hrs hpos
  obtain ⟨hE, hB⟩ := ofLoop_answered_epsBeta Masked.dataE
    (withFiles (withLink (withInner Masked.dataE L₁ real (finHonest (ofLoopLink L₁ src real))) src) src) real x w hSv
    hR (ν, t) (m₀, ρ₀) ops ha hne rfl hxs hq1 hloc hreal hrs hpos (by omega)
  have hon := ofLoop_answered_on Masked.dataE
    (withFiles (withLink (withInner Masked.dataE L₁ real (finHonest (ofLoopLink L₁ src real))) src) src) real x
    (ν, t) (m₀, ρ₀) ops ha (by simp [hin'])
  have hco : (ofLoopLink (withInner Masked.dataE L₁ real (finHonest (ofLoopLink L₁ src real))) src real).coins x
      ((m₀, ρ₀) :: ops).length = loopNr x (q + 1) := by
    rw [List.length_cons, hq1]
    rfl
  have hcfg : (ofLoopLink (withInner Masked.dataE L₁ real (finHonest (ofLoopLink L₁ src real))) src real).cfg x
      (ν, t) ((m₀, ρ₀) :: ops) = (withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops) := by
    unfold cfg drawsAt
    rfl
  have hrow' : ((withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).rowH.size < 2 ^ 64 := by
    rw [← hcfg]
    exact hrow
  have hn : loopNr x (q + 1) =
      [1, _root_.Flock.Firewall.Session.INNER_CONS, 1, _root_.Flock.Zk.Q].getD (j + 1 - rsched.size) 0 := by
    rw [hslot]
    exact hnr _ he4
  have hnf : loopNr x (q + 1) < 2 ^ 64 ∧ loopNr x (q + 1) ≠ 0 ∧
      (j + 1 - rsched.size = 2 → loopNr x (q + 1) = 1) := by
    rw [hn]
    generalize j + 1 - rsched.size = e at he4 ⊢
    obtain rfl | rfl | rfl | rfl : e = 0 ∨ e = 1 ∨ e = 2 ∨ e = 3 := by omega
    all_goals decide
  have hR0 : ∀ vals,
      (ofLoopLink (withInner Masked.dataE L₁ real (finHonest (ofLoopLink L₁ src real))) src real).scheduled x w
        (ν, t) ((m₀, ρ₀) :: ops) vals =
      (innerSched ((withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops)) { m with stopped := false }
        (finHonest (ofLoopLink L₁ src real) x w (ν, t) ((m₀, ρ₀) :: ops)) (loopNr x (q + 1))
        (j + 1 - rsched.size)).data.toList := by
    intro vals
    rw [hsc vals]
    exact congrArg (fun b : ByteArray => b.data.toList)
      (Loop.innerSched_state_congr _ _ _ he4 (fun _ => hE.symm) (fun h3 => (hB (by omega)).symm))
  have hinE : m.inner = some (j + 1 - rsched.size) := by
    rw [hin']
    congr 1
    omega
  refine Loop.roundOk_ownAt finS.honestVals _ ha hcfg hco (hR0 _) (by omega) he4 hon hinE hrow' hnf.1 hnf.2.1
    hnf.2.2 (fun _ => ?_) hcr
  have hrep : (ofLoopLink L₁ src real).rep x ((m₀, ρ₀) :: ops).length = 2 * jf.val + rf.val := by
    show Loop.repOf x (ops.length + 1) = _
    rw [show ops.length + 1 = q + 2 by omega, repOf_add_two hxs, hloc]
    exact hkjr
  rw [innerSched_finHonest2 (ofLoopLink L₁ src real) x w (ν, t) ((m₀, ρ₀) :: ops) _ _ _ hrep,
    vsTauAt_ofLoopLink_withInner]
  exact setTau_inner2_self m _

end LeanRelease

end Claims.ZKServer
