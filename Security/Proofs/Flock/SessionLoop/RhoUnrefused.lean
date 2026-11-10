module

public import Proofs.Flock.SessionLoop.PcsRank

/-!
# A rep's `ρ₀, ρ₁` at its `ρ₁` coins

Past a rep's ring switch its schedule is the opening's binding and level 0's bodies (`Bind.sched_l0`), and the frame
moves through them: the binding starts level 0's masked phase at the algebraic phase's last position, each body absorbs
its values, `ρ₀`'s coins after the last are the frame's first `ρ` words (`Masker.answer`), and at `ρ₁`'s coins the frame
refuses words dependent on them.

* **`Bind.sched_gamma`**: a rep's schedule fed past `γ`'s coins and into level 0's holds the zerocheck's head, the
  opening's label and quiet rounds up to `γ`'s.
* **`RhoKeep.absorb_rho`**, **`RhoKeep.label_rho`**, **`RhoKeep.roundFree_rho`**: an absorbed value, a label and a
  round keep the frame's `ρ` words.
* **`RhoKeep.pair_rho`**, **`RhoKeep.sched_rhoNil`**: so up to `γ`'s round, where the flags hold, the frame has no `ρ`
  word.
* **`RhoKeep.L0Inv`**, **`RhoKeep.bind_step`**, **`RhoKeep.l0_step`**: the frame inside level 0 after its body `b`,
  from the binding on.
* **`RhoKeep.RhoInv`**, **`RhoKeep.rho_step`**: after `ρ₀`'s coins and `ρ₁`'s round, the frame holds `ρ₀`'s words.
* **`RhoKeep.sched_l0Run`**, **`RhoKeep.repState_rho`**: along a rep's schedule, the frame after level 0's bodies, and
  at `ρ₁`'s coins the pair is independent.
* **`LeanRelease.rho_unrefused_sched_pf`**: `LeanRelease.rho_unrefused_sched`.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer.Bind

open _root_.Flock (F128 Transcript TMW)
open _root_.Flock.Firewall.Frame (Cfg Masker)
open _root_.Flock.Firewall.Drive (Step OpenM stepAsks openStep)
open _root_.Flock.Firewall.Finish (repOpening repPiop commitFw claimsOf ringSwitchP_eq switchVec)
open FlockSoundness.Refine (Frame encs)
open FlockComplete (PS asksCut ringSwitchCut)
open FlockComplete.Replay (bind_go liftTM_ok)
open FlockComplete.LigLock.Fed (feedR sentQ pendC feed_eq CoinAt repOpening_feed cutAsks cutAsks_toC openStep_sent)
open Switch (rsBody)
open Loop

/-- **A rep's schedule fed into level 0**: fed the coins of the algebraic phase's rounds, the ring switch's, `γ`'s and
up to level 0's first `j - 1`, a rep's schedule holds the zerocheck's head at round `0`, the opening's label at its
first round and quiet rounds up to `γ`'s. -/
theorem sched_gamma {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} {cr : Array (Array F128)} {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr = .ok sched) (hm14 : 14 ≤ t.st.m) (hdom : dom.size < 2 ^ 64)
    (hdg : t.st.digest.size < 2 ^ 64) (hcap : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64)
    (hm : t.st.m < 2 ^ 64) (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (hx : xs.size + 2 < 2 ^ 64)
    (hk : 1 ≤ t.sch.k0) (hn : t.sch.n < 2 ^ 62) (hq : ∀ i, (t.sch.levels.getD i default).q < 2 ^ 63)
    (hn0 : t.sch.n ≠ 0) {j : ℕ} (hj : j ≤ t.sch.k0 + 4) (hcr : cr.size = alg.size + (xs.size + 2) + j) :
    sched.size = alg.size + (xs.size + 2) + j + 1 ∧ 1 < alg.size ∧
      sched.getD 0 (.empty, 0) = (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8,
        .bytes t.st.digest, .bytes (_root_.Flock.Firewall.Finish.flat t.cm0.cap),
        .label _root_.Flock.Firewall.Frame.LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray, 6) ∧
      Flags.PcsRound (sched.getD alg.size (.empty, 0)) ∧
      ∀ k, 1 ≤ k → k ≤ alg.size + (xs.size + 2) → k ≠ alg.size → Quiet.QRound (sched.getD k (.empty, 0)) := by
  obtain ⟨rest, hrest⟩ := schedSoFar_pre hal hs
  have hAle : alg.size ≤ sched.size := by rw [hrest, Array.size_append]; omega
  obtain ⟨hdrop, hnf⟩ := schedSoFar_sentQ hal hs (by omega)
  have hcsl : (cr.extract alg.size cr.size).toList.length = xs.size + 2 + j := by
    simp only [Array.length_toList, Array.size_extract]
    omega
  have hlenF := feed_part _ _ _
    (Count.openStep_end t rep dom xs hi (cr.extract 0 alg.size) hdom hdg hcap hm hx hk hn hq)
    (by rw [hcsl]; have := openList_len_l0 t.sch (N := xs.size + 2) (by omega) hn0; omega) hnf
  rw [hcsl] at hlenF
  have hsz : sched.size = alg.size + (xs.size + 2) + j + 1 := by
    have := congrArg List.length hdrop
    simp only [List.length_drop, List.length_map, Array.length_toList] at this
    omega
  obtain ⟨sv, hsv, hbody⟩ := openStep_switch t rep dom xs hi (cr.extract 0 alg.size) hdom hdg hcap hm hmk
    (cr.extract alg.size cr.size).toList (by omega)
  have hB : ∀ i, i ≤ xs.size + 2 →
      (sched.getD (alg.size + i) (.empty, 0)).1.data.toList = rsBody sv (xs.size + 2) [] i :=
    fun i hi' => sched_bytes hdrop hbody (by omega) hi'
  have hC : ∀ i, i ≤ xs.size + 2 →
      (sched.getD (alg.size + i) (.empty, 0)).2 = (Count.openList (xs.size + 2) t.sch).getD i 0 :=
    fun i hi' => sched_count hal hs hdom hdg hcap hm hx hk hn hq (by omega)
  obtain ⟨ha0, ha1⟩ := Sched.rounds_two hm14 hm hal
  have hg : ∀ k, k < alg.size → sched.getD k (.empty, 0) = (alg[k]?).getD (.empty, 0) := fun k hk' => by
    rw [hrest, Array.getD_eq_getD_getElem?, Array.getElem?_append_left hk']
  have h1s : 1 < alg.size := (Array.getElem?_eq_some_iff.1 ha1).1
  have h0 : sched.getD 0 (.empty, 0) = (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8,
      .bytes t.st.digest, .bytes (_root_.Flock.Firewall.Finish.flat t.cm0.cap),
      .label _root_.Flock.Firewall.Frame.LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray, 6) := by
    rw [hg 0 (by omega), ha0, Option.getD_some, head_encs]
  have hN0 : xs.size + 2 ≠ 0 := by omega
  have hsv' : ∀ i, (sv i).size < 2 ^ 64 := fun i => by rw [hsv]; decide
  have hp : Flags.PcsRound (sched.getD alg.size (.empty, 0)) := by
    have h7 := hC 0 (by omega)
    rw [openList_lt t.sch hN0 (by omega), Nat.add_zero] at h7
    have hb := hB 0 (by omega)
    rw [Nat.add_zero] at hb
    rw [show sched.getD alg.size (.empty, 0) = ((sched.getD alg.size (.empty, 0)).1, 7) from Prod.ext rfl h7]
    exact Switch.pcsRound_rs sv hN0 (hsv' 0) hb
  have hq' : ∀ k, 1 ≤ k → k ≤ alg.size + (xs.size + 2) → k ≠ alg.size →
      Quiet.QRound (sched.getD k (.empty, 0)) := by
    intro k h1 hkJ hkA
    by_cases hkl : k < alg.size
    · rw [hg k hkl, Array.getElem?_eq_getElem hkl, Option.getD_some]
      exact Quiet.rounds_quiet hm hdg hcap hdom hal k _ h1 (Array.getElem?_eq_getElem hkl)
    · obtain ⟨i, rfl⟩ : ∃ i, k = alg.size + i := ⟨k - alg.size, by omega⟩
      have hcnt := hC i (by omega)
      have hb := hB i (by omega)
      by_cases hiN : i < xs.size + 2
      · rw [openList_lt t.sch hN0 hiN] at hcnt
        rw [show sched.getD (alg.size + i) (.empty, 0) = ((sched.getD (alg.size + i) (.empty, 0)).1, 7) from
          Prod.ext rfl hcnt]
        exact Switch.qRound_rs sv (by omega) hiN (hsv' i) hb
      · obtain rfl : i = xs.size + 2 := by omega
        rw [openList_at t.sch hN0] at hcnt
        rw [show sched.getD (alg.size + (xs.size + 2)) (.empty, 0) =
          ((sched.getD (alg.size + (xs.size + 2)) (.empty, 0)).1, xs.size + 2) from Prod.ext rfl hcnt]
        exact Switch.qRound_gamma sv hN0 hx hb
  exact ⟨hsz, h1s, h0, hp, hq'⟩

end Claims.ZKServer.Bind

namespace Claims.ZKServer.RhoKeep

open _root_.Flock (V F128)
open _root_.Flock.Firewall.Frame (Cfg Masker MaskSlot Draws roundFree isLabel LABEL_PCS LABEL_ZC claimPoints
  INNER_ROUNDS)
open FlockSoundness.Refine (Frame encs)
open Loop

/-! ## Rounds keep the `ρ` words -/

/-- An absorbed value keeps the frame's `ρ` words. -/
theorem absorb_rho {c : Cfg} {s : Masker} {v : F128} {q : Masker × F128} (h : s.absorb c v = .ok q) :
    q.1.rho = s.rho := by
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

/-- A label keeps the frame's `ρ` words. -/
theorem label_rho {c : Cfg} {s : Masker} {l : ByteArray} {q : Masker × ByteArray} (h : s.label c l = .ok q) :
    q.1.rho = s.rho := by
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

/-- A round keeps the frame's `ρ` words, whatever its bytes. -/
theorem roundFree_rho {c : Cfg} {s : Masker} {content : ByteArray} {n : ℕ} {q : ByteArray × Masker × Array ℕ}
    (h : roundFree c s content n = .ok q) : q.2.1.rho = s.rho := by
  unfold roundFree at h
  by_cases hI : (s.inner == some INNER_ROUNDS) = true
  · simp only [hI, ↓reduceIte] at h
    cases h
  simp only [hI, Bool.false_eq_true, ↓reduceIte, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at h
  obtain ⟨r, hr, h⟩ := bindV_ok h
  have hQ := Ranked.forIn_inv (fun x : Option (ByteArray × Masker × Array ℕ) × Masker × ByteArray × Array ℕ × ℕ =>
      x.2.1.rho = s.rho ∧ ∀ v, x.1 = some v → v.2.1.rho = s.rho) _ ?step _ _ _
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
      exact ⟨(label_rho hx).trans hk, fun _ hv => by cases hv⟩
    · have hy := Ranked.ite_bind_ok hy
      generalize (_root_.Flock.u64OfLE content (pos + 8)).toNat = len at hy
      generalize content.get! (pos + 1) = kind at hy
      by_cases h3 : (!(kind == 2 || kind == 1 && len == 1)) = true
      · rw [ite_eq_left h3] at hy
        cases hy
      rw [ite_eq_right h3] at hy
      obtain ⟨r', hr', hy⟩ := bindV_ok hy
      cases hy
      have hQ' : r'.1.rho = s.rho := by
        refine Ranked.forIn_inv (fun x : Masker × ByteArray × Array ℕ => x.1.rho = s.rho) _ ?_ _ _ _ hk hr'
        intro i x y hx hy
        by_cases h8 : (x.1.on && decide (x.1.pos < c.nPiop)) = true
        · rw [ite_eq_left h8] at hy
          obtain ⟨a, ha, hy⟩ := bindV_ok hy
          cases hy
          exact (absorb_rho ha).trans hx
        · rw [ite_eq_right h8] at hy
          obtain ⟨a, ha, hy⟩ := bindV_ok hy
          cases hy
          exact (absorb_rho ha).trans hx
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
        exact ⟨hk, fun v hv => by cases hv; exact hk⟩
      · cases hy
        exact ⟨hk, fun v hv => by cases hv; exact hk⟩
    · have hy := Ranked.ite_bind_ok hy
      cases hy
      exact ⟨hk, fun _ hv => by cases hv⟩
    · cases hy

theorem getElem_two (a b : F128) : #[a, b][0]! = a ∧ #[a, b][1]! = b := by
  constructor <;> first | rfl | simp

/-! ## Up to `γ`'s round -/

/-- **A scheduled round after the coins keeps the `ρ` words**, from flags after round `k`, where the rep runs on. -/
theorem pair_rho {c : Cfg} {A k : ℕ} {S : _root_.Flock.Firewall.Frame.State} {cs : Array F128}
    {r : ByteArray × ℕ} (hI : Flags.FInv c A k S.m)
    (h2 : (_root_.Flock.Firewall.Frame.run c S [.coins (.ok cs), .round r.2 r.1 (.sched r.1)]).2 = false) :
    (_root_.Flock.Firewall.Frame.run c S [.coins (.ok cs), .round r.2 r.1 (.sched r.1)]).1.m.rho = S.m.rho := by
  obtain ⟨s1, ha, he⟩ := run_coins_cons c S cs _ h2
  rw [he] at h2 ⊢
  have hF : Flags.Fl c A k S.m := by
    rcases hI with ho | ⟨-, hF⟩
    · exact absurd ha (answer_owed_none _ ho)
    · exact hF
  have e := answer_alg c S.m cs hF.2.2.1 hF.1 ha
  have ho1 : s1.owed = none := by subst e; rfl
  have hr1 : s1.rho = S.m.rho := by subst e; rfl
  rcases Loop.run_sched_cons (st := { S with m := s1 }) ho1 h2 with ⟨out, sx, why, hck, he'⟩ | he'
  · rw [he']
    show sx.rho = S.m.rho
    rcases Loop.check_self hck with ⟨fr, hr⟩ | ⟨fr, hr⟩
    · exact (roundFree_rho hr).trans hr1
    · exact (roundFree_rho hr).trans hr1
  · rw [he']
    exact hr1

/-- **No `ρ` word before level 0**: from the rep's start, past the zerocheck's head and rounds `1 … J`, quiet but for
the opening's first at `A`, each after the coins of the one before, where the rep runs on, the frame has no `ρ`
word. -/
theorem sched_rhoNil {c : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) (P : Array F128)
    (sched : Array (ByteArray × ℕ)) (cr : Array (Array F128)) {digest cap dom : ByteArray}
    (hdg : digest.size < 2 ^ 64) (hcap : cap.size < 2 ^ 64) (hdom : dom.size < 2 ^ 64)
    (h0 : sched.getD 0 (.empty, 0) = (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes digest,
      .bytes cap, .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray, 6))
    {A : ℕ} (hA0 : 0 < A) (hp : Flags.PcsRound (sched.getD A (.empty, 0))) :
    ∀ J, (∀ k, 1 ≤ k → k ≤ J → k ≠ A → Quiet.QRound (sched.getD k (.empty, 0))) →
      (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr J)).2 = false →
      (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr J)).1.m.rho = #[]
  | 0, _, h2 => by
    rw [Flags.schedReqs_zero] at h2 ⊢
    obtain ⟨p, -, -, he⟩ := Ranked.run_prior_eq c P _ h2
    rw [he] at h2 ⊢
    rcases Loop.run_sched_cons (st := { m := { prior := p } }) rfl h2 with ⟨out, sx, why, hck, he'⟩ | he'
    · rw [he']
      show sx.rho = #[]
      rcases Loop.check_self hck with ⟨fr, hr⟩ | ⟨fr, hr⟩
      · exact roundFree_rho hr
      · exact roundFree_rho hr
    · rw [he']
      rfl
  | J + 1, hq, h2 => by
    rw [schedReqs_succ, Loop.run_append] at h2 ⊢
    cases hb : (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr J)).2
    · simp only [hb, Bool.false_eq_true, ↓reduceIte] at h2 ⊢
      have hq' : ∀ k, 1 ≤ k → k ≤ J → k ≠ A → Quiet.QRound (sched.getD k (.empty, 0)) :=
        fun k h1 hk hA => hq k h1 (by omega) hA
      rw [pair_rho (Flags.sched_flags hc P sched cr hdg hcap hdom h0 hA0 hp J hq' hb) h2]
      exact sched_rhoNil hc P sched cr hdg hcap hdom h0 hA0 hp J hq' hb
    · simp [hb] at h2

/-! ## Level 0 -/

/-- **The frame after level 0's body `b`** (the binding at `b = 0`), owing its `n` coins: in level 0's masked phase at
the next body's first value, with no `ρ` word yet; or it owes nothing (the round refused). -/
def L0Inv (c : Cfg) (b n : ℕ) (s : Masker) : Prop :=
  s.owed = none ∨ (s.owed = some n ∧ s.done = true ∧ s.on = true ∧ s.l0 = true ∧ s.doneL0 = false ∧
    s.inner = none ∧ s.pos = c.nPiop + l0Off (b + 1) ∧ s.rho = #[])

/-- **The binding**: after `γ`'s round, past the opening's label, with no `ρ` word, `γ`'s coins and the binding with
`z`'s squeeze start level 0 at the algebraic phase's last position. -/
theorem bind_step {c : Cfg} {A k : ℕ} {S : _root_.Flock.Firewall.Frame.State} {cs : Array F128}
    {r : ByteArray × ℕ} {T : F128} {cap : ByteArray} (hI : Flags.FInv c A k S.m) (hAk : A ≤ k)
    (hrho : S.m.rho = #[]) (hlig : c.lig = true) (hcap : cap.size < 2 ^ 64)
    (hr : r.1.data.toList = encs (openBind T cap) ++ (Frame.sq 2 r.2).enc) (hn0 : r.2 ≠ 0) (hn : r.2 < 2 ^ 64)
    (h2 : (_root_.Flock.Firewall.Frame.run c S [.coins (.ok cs), .round r.2 r.1 (.sched r.1)]).2 = false) :
    L0Inv c 0 r.2 (_root_.Flock.Firewall.Frame.run c S [.coins (.ok cs), .round r.2 r.1 (.sched r.1)]).1.m := by
  obtain ⟨s1, ha, he⟩ := run_coins_cons c S cs _ h2
  rw [he] at h2 ⊢
  have hF : Flags.Fl c A k S.m := by
    rcases hI with ho | ⟨-, hF⟩
    · exact absurd ha (answer_owed_none _ ho)
    · exact hF
  have e := answer_alg c S.m cs hF.2.2.1 hF.1 ha
  have hr1 : s1.rho = #[] := by subst e; exact hrho
  obtain ⟨hF1, ho1⟩ := Flags.flags_answer hF ha
  obtain ⟨a1, a2, a3, -, a5⟩ := hF1
  obtain ⟨b1, b2, b3⟩ := a5 hAk
  rcases Loop.run_sched_cons (st := { S with m := s1 }) ho1 h2 with ⟨out, sx, why, hck, he'⟩ | he'
  · rw [he']
    show L0Inv c 0 r.2 sx
    have key : ∀ (s0 : Masker) (fr : Array ℕ), s0.on = false → s0.done = true → s0.doneL0 = false →
        s0.inner = none → s0.pos = c.nPiop → s0.rho = #[] →
        roundFree c s0 r.1 r.2 = .ok (out, sx, fr) → L0Inv c 0 r.2 sx := by
      intro s0 fr q1 q2 q3 q4 q5 q6 hr'
      rw [Flags.eq_mk_encs hr] at hr'
      obtain ⟨out', h', -⟩ := roundFree_openBind c s0 T cap r.2 q1 q2 hlig q3 q4 hcap hn0 hn
      rw [h'] at hr'
      simp only [Except.ok.injEq, Prod.mk.injEq] at hr'
      obtain ⟨-, rfl, -⟩ := hr'
      refine Or.inr ⟨rfl, q2, rfl, rfl, q3, q4, ?_, q6⟩
      show s0.pos = _
      rw [q5, show l0Off (0 + 1) = 0 from rfl, Nat.add_zero]
    rcases Loop.check_self hck with ⟨fr, hr'⟩ | ⟨fr, hr'⟩
    · exact key s1 fr b2 b1 a2 a3 b3 hr1 hr'
    · exact key { s1 with stopped := true } fr b2 b1 a2 a3 b3 hr1 hr'
  · rw [he']
    exact Or.inl ho1

/-- **Level 0's body `b + 1`**: after body `b`, its coins and body `b + 1`, frames `G` absorbing `l0N (b + 1)` values
inside the layout with its squeeze, keep the frame in level 0. -/
theorem l0_step {c : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) {b n : ℕ}
    {S : _root_.Flock.Firewall.Frame.State} {cs : Array F128} {r : ByteArray × ℕ} (hI : L0Inv c b n S.m)
    (hsz : c.nPiop + l0Off (b + 1) + l0N (b + 1) ≤ c.order.size) {G : List Frame} (hm : ∀ f ∈ G, midFrame f)
    (hp : ∀ f ∈ G, plainFrame f) (habs : nAbs G = l0N (b + 1))
    (hr : r.1.data.toList = encs G ++ (Frame.sq (l0K (b + 1)) (l0Nc (b + 1))).enc) (hr2 : r.2 = l0Nc (b + 1))
    (h2 : (_root_.Flock.Firewall.Frame.run c S [.coins (.ok cs), .round r.2 r.1 (.sched r.1)]).2 = false) :
    L0Inv c (b + 1) r.2 (_root_.Flock.Firewall.Frame.run c S [.coins (.ok cs), .round r.2 r.1 (.sched r.1)]).1.m := by
  obtain ⟨s1, ha, he⟩ := run_coins_cons c S cs _ h2
  rw [he] at h2 ⊢
  rcases hI with ho | ⟨ho, hd, hon, hl, hdl, hin, hps, hrh⟩
  · exact absurd ha (answer_owed_none _ ho)
  have hcs := answer_size ho ha
  have hN : 1 ≤ l0N (b + 1) := by unfold l0N; split_ifs <;> omega
  have e : s1 = { S.m with owed := none } := by
    rw [answer_l0_mid c S.m cs (by rw [ho, hcs]) hin hl (by omega)] at ha
    exact (Except.ok.inj ha).symm
  subst e
  rcases Loop.run_sched_cons (st := { S with m := { S.m with owed := none } }) rfl h2 with
    ⟨out, sx, why, hck, he'⟩ | he'
  · rw [he']
    show L0Inv c (b + 1) r.2 sx
    obtain ⟨hk1, hk2, hk3⟩ := l0K_ok (b + 1)
    have key : ∀ (s0 : Masker) (fr : Array ℕ), s0.on = true → s0.inner = none → s0.pos = S.m.pos →
        s0.done = S.m.done → s0.l0 = S.m.l0 → s0.doneL0 = S.m.doneL0 → s0.rho = S.m.rho →
        roundFree c s0 r.1 r.2 = .ok (out, sx, fr) → L0Inv c (b + 1) r.2 sx := by
      intro s0 fr q1 q2 q3 q4 q5 q6 q7 hr'
      rw [Flags.eq_mk_encs hr, hr2] at hr'
      obtain ⟨out', fr', h', -⟩ := roundFree_on c s0 G (l0K (b + 1)) (l0Nc (b + 1)) hc q1 q2 hm hp hk1 hk2 hk3
        (by rw [q3, hps, habs]; exact hsz)
      rw [h'] at hr'
      simp only [Except.ok.injEq, Prod.mk.injEq] at hr'
      obtain ⟨-, rfl, -⟩ := hr'
      refine Or.inr ⟨congrArg some hr2.symm, q4.trans hd, q1, q5.trans hl, q6.trans hdl, q2, ?_, q7.trans hrh⟩
      show s0.pos + nAbs G = _
      rw [q3, hps, habs, Nat.add_assoc c.nPiop, l0Off_succ (i := b + 1) (by omega)]
    rcases Loop.check_self hck with ⟨fr, hr'⟩ | ⟨fr, hr'⟩
    · exact key { S.m with owed := none } fr hon hin rfl rfl rfl rfl rfl hr'
    · exact key { S.m with owed := none, stopped := true } fr hon hin rfl rfl rfl rfl rfl hr'
  · rw [he']
    exact Or.inl rfl

/-- **The frame holding `ρ₀`'s words** `r0, r1` at `ρ₁`'s round, owing its two coins, past level 0's last masked value;
or it owes nothing (the round refused). -/
def RhoInv (c : Cfg) (r0 r1 : F128) (s : Masker) : Prop :=
  s.owed = none ∨ (s.owed = some 2 ∧ s.l0 = true ∧ s.on = true ∧ s.inner = none ∧ s.pos = c.order.size ∧
    s.rho = #[r0, r1])

/-- **`ρ₀`'s coins and `ρ₁`'s round**: after level 0's last masked value, owing `ρ₀`'s two coins, its coins and `ρ₁`'s
squeeze leave the frame holding them as its `ρ` words. -/
theorem rho_step {c : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) {b : ℕ}
    {S : _root_.Flock.Firewall.Frame.State} {cs : Array F128} {r : ByteArray × ℕ} (hI : L0Inv c b 2 S.m)
    (hsz : c.nPiop + l0Off (b + 1) = c.order.size) (hr : r.1.data.toList = (Frame.sq 2 2).enc) (hr2 : r.2 = 2)
    (h2 : (_root_.Flock.Firewall.Frame.run c S [.coins (.ok cs), .round r.2 r.1 (.sched r.1)]).2 = false) :
    RhoInv c cs[0]! cs[1]! (_root_.Flock.Firewall.Frame.run c S [.coins (.ok cs), .round r.2 r.1 (.sched r.1)]).1.m := by
  obtain ⟨s1, ha, he⟩ := run_coins_cons c S cs _ h2
  rw [he] at h2 ⊢
  rcases hI with ho | ⟨ho, hd, hon, hl, hdl, hin, hps, hrh⟩
  · exact absurd ha (answer_owed_none _ ho)
  obtain ⟨a, a', rfl⟩ := FlockComplete.JsonRT.two_of (answer_size ho ha)
  obtain ⟨g0, g1⟩ := getElem_two a a'
  rw [g0, g1]
  have e : s1 = { S.m with owed := none, rho := #[a, a'] } := by
    rw [answer_rho_first c S.m a a' ho hin hl (hps.trans hsz) hrh] at ha
    exact (Except.ok.inj ha).symm
  subst e
  rcases Loop.run_sched_cons (st := { S with m := { S.m with owed := none, rho := #[a, a'] } }) rfl h2 with
    ⟨out, sx, why, hck, he'⟩ | he'
  · rw [he']
    show RhoInv c a a' sx
    have key : ∀ (s0 : Masker) (fr : Array ℕ), s0.on = true → s0.inner = none → s0.pos = c.order.size →
        s0.l0 = true → s0.rho = #[a, a'] → roundFree c s0 r.1 r.2 = .ok (out, sx, fr) → RhoInv c a a' sx := by
      intro s0 fr q1 q2 q3 q4 q5 hr'
      rw [Flags.eq_mk_encs (l := encs [] ++ (Frame.sq 2 2).enc) (hr.trans rfl), hr2] at hr'
      obtain ⟨out', fr', h', -⟩ := roundFree_on c s0 [] 2 2 hc q1 q2 (by simp) (by simp) (.inl rfl) (by decide)
        (by norm_num) (by rw [q3]; simp [nAbs])
      rw [h'] at hr'
      simp only [Except.ok.injEq, Prod.mk.injEq] at hr'
      obtain ⟨-, rfl, -⟩ := hr'
      refine Or.inr ⟨rfl, q4, q1, q2, ?_, q5⟩
      show s0.pos + nAbs [] = _
      rw [q3]
      rfl
    rcases Loop.check_self hck with ⟨fr, hr'⟩ | ⟨fr, hr'⟩
    · exact key { S.m with owed := none, rho := #[a, a'] } fr hon hin (hps.trans hsz) hl rfl hr'
    · exact key { S.m with owed := none, rho := #[a, a'], stopped := true } fr hon hin (hps.trans hsz) hl rfl hr'
  · rw [he']
    exact Or.inl rfl

/-! ## Along a rep's schedule -/

/-- **A rep's frame through level 0**: where a rep's schedule is fed the coins of its algebraic phase's rounds, the
ring switch's, `γ`'s and level 0's first `j`, the frame after the binding and each body `b ≤ j` up to `k₀ + 2`, where the
rep runs on, is in level 0 owing the body's count (`L0Inv`). -/
theorem sched_l0Run {c : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size)
    {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} {cr : Array (Array F128)} {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr.pop = .ok sched) (hm14 : 14 ≤ t.st.m) (hdg : t.st.digest.size < 2 ^ 64)
    (hcap : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64) (hdom : dom.size < 2 ^ 64)
    (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (hx : xs.size + 2 < 2 ^ 64) {tp : ℕ}
    (hz : _root_.Flock.Zk.schedule t.st.m = .ok (t.sch, tp)) {lay : _root_.Flock.Zk.PadLayout} {d : Draws}
    {mk : Option MaskSlot} (hlay : lay = _root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0)
    (hcd : c = Cfg.ofDraws lay d mk) {j : ℕ} (hj : j ≤ t.sch.k0 + 3)
    (hcr : cr.pop.size = alg.size + (xs.size + 2) + 1 + j) (P : Array F128) :
    sched.size = alg.size + (xs.size + 2) + 1 + j + 1 ∧ ∀ b, b ≤ j → b ≤ t.sch.k0 + 2 →
      (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr (alg.size + (xs.size + 2) + 1 + b))).2 = false →
      L0Inv c b (if b = 0 then t.sch.n else l0Nc b)
        (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr (alg.size + (xs.size + 2) + 1 + b))).1.m := by
  obtain ⟨hm36, hk, hn, hq⟩ := Count.sched_facts hz
  have hn0 : t.sch.n ≠ 0 := by have := (Count.sched_rho hz).1; omega
  have hm : t.st.m < 2 ^ 64 := by omega
  obtain ⟨hsz, h1s, h0, hp, hq'⟩ := Bind.sched_gamma hal hs hm14 hdom hdg hcap hm hmk hx hk hn hq hn0
    (j := 1 + j) (by omega) (by rw [hcr]; omega)
  obtain ⟨-, T, sws, -, hbody⟩ := Bind.sched_l0 hal hs hdom hdg hcap hm hmk hx hk hn hq hn0 hm14 (j := j)
    (by omega) hcr
  refine ⟨by omega, ?_⟩
  have hlk : lay.k0 = t.sch.k0 := by subst hlay; rfl
  have hood : lay.nOod0 = 1 := by subst hlay; rfl
  have hk' : 0 < lay.k0 := by omega
  have hlig : c.lig = true := by rw [hcd, Loop.ofDraws_lig, decide_eq_true_eq]; omega
  have hpi : c.nPiop = absPos lay (lay.lgOod 0) := by rw [hcd]; exact nPiop_ood lay d mk hk'
  have hord : c.order.size = (_root_.Flock.Firewall.Frame.absorbed lay).size := by subst hcd; rfl
  intro b
  induction b with
  | zero =>
    intro _ _ h2
    rw [if_pos rfl]
    rw [Nat.add_zero, schedReqs_succ, Loop.run_append] at h2 ⊢
    cases hb : (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr (alg.size + (xs.size + 2)))).2
    · simp only [hb, Bool.false_eq_true, ↓reduceIte] at h2 ⊢
      have hI := Flags.sched_flags hc P sched cr hdg hcap hdom h0 (by omega) hp _ hq' hb
      have hR := sched_rhoNil hc P sched cr hdg hcap hdom h0 (by omega) hp _ hq' hb
      have hr2 : (sched.getD (alg.size + (xs.size + 2) + 1) (.empty, 0)).2 = t.sch.n := by
        have h1 := (hbody 0 (Nat.zero_le _)).2
        rw [List.getD_eq_getElem?_getD,
          Count.openList_l0 t.sch (N := xs.size + 2) (by omega) hn0 (i := 0) (Nat.zero_le _)] at h1
        simpa using h1
      have hr1 : (sched.getD (alg.size + (xs.size + 2) + 1) (.empty, 0)).1.data.toList =
          encs (openBind T (FlockComplete.flat t.cm0.toC.cap)) ++
            (Frame.sq 2 (sched.getD (alg.size + (xs.size + 2) + 1) (.empty, 0)).2).enc := by
        have h1 := (hbody 0 (Nat.zero_le _)).1
        rw [Nat.add_zero, FlockComplete.LigLock.l0BodyAt_zero, List.nil_append] at h1
        rw [h1, hr2, openBind]
      have key := bind_step (r := sched.getD (alg.size + (xs.size + 2) + 1) (.empty, 0))
        (cap := FlockComplete.flat t.cm0.toC.cap) hI (by omega) hR hlig hcap hr1 (by rw [hr2]; exact hn0)
        (by rw [hr2]; omega) h2
      rw [show t.sch.n = (sched.getD (alg.size + (xs.size + 2) + 1) (.empty, 0)).2 from hr2.symm]
      exact key
    · simp [hb] at h2
  | succ b ih =>
    intro hbj hbk h2
    rw [if_neg (show b + 1 ≠ 0 by omega)]
    rw [show alg.size + (xs.size + 2) + 1 + (b + 1) = alg.size + (xs.size + 2) + 1 + b + 1 by omega,
      schedReqs_succ, Loop.run_append] at h2 ⊢
    cases hb : (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr (alg.size + (xs.size + 2) + 1 + b))).2
    · simp only [hb, Bool.false_eq_true, ↓reduceIte] at h2 ⊢
      have hI := ih (by omega) (by omega) hb
      obtain ⟨hb1, hc1⟩ := hbody (b + 1) hbj
      rw [show alg.size + (xs.size + 2) + 1 + (b + 1) = alg.size + (xs.size + 2) + 1 + b + 1 by omega,
        ← hlay] at hb1
      rw [show alg.size + (xs.size + 2) + 1 + (b + 1) = alg.size + (xs.size + 2) + 1 + b + 1 by omega,
        List.getD_eq_getElem?_getD,
        Count.openList_l0 t.sch (N := xs.size + 2) (by omega) hn0 (i := b + 1) (by omega),
        if_neg (show b + 1 ≠ 0 by omega), Option.getD_some] at hc1
      obtain ⟨G, H, R, hBz, -⟩ := FlockComplete.LigLock.l0Sent_at
        (coin := fun j i => ((cr.pop.extract alg.size cr.pop.size).toList.getD j #[])[i]!)
        (p0 := xs.size + 2 + 1) (P := []) (T := T) (cap := FlockComplete.flat t.cm0.toC.cap) (sch := t.sch)
        (t := t.g) (b := FlockComplete.basisVals sws (t.st.m - 7)) (lay := lay) (rep := rep) (extra := t.extra)
        (nonce := fun _ _ => 0) (d := d) (mk := mk) hk' hood hlk.symm lay.len (i := b + 1) (by omega) (by omega)
      have hsz' : c.nPiop + l0Off (b + 1) + l0N (b + 1) ≤ c.order.size := by
        have h1 := l0_size lay hk' hood (i := b + 1) (by omega) (by omega)
        rw [hpi, hord]
        omega
      rw [← hc1]
      exact l0_step hc hI hsz' (G := G) R.mid_G R.plain_G R.abs_G (hb1.trans hBz) hc1 h2
    · simp [hb] at h2

/-- **At a rep's `ρ₁` coins its pair is independent**: where the loop's state at a rep's round past `ρ₁`'s takes the
round's coins `cs` at a layout of the rep's size, `ρ₀`'s coins and `cs` are independent. -/
theorem repState_rho {c c'' : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size)
    {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} {cr0 cr : Array (Array F128)} {ns : ℕ → ℕ} {alg : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hm14 : 14 ≤ t.st.m) (hdg : t.st.digest.size < 2 ^ 64)
    (hcap : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64) (hdom : dom.size < 2 ^ 64)
    (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (hx : xs.size + 2 < 2 ^ 64) {tp : ℕ}
    (hz : _root_.Flock.Zk.schedule t.st.m = .ok (t.sch, tp)) {lay : _root_.Flock.Zk.PadLayout} {d : Draws}
    {mk : Option MaskSlot} (hlay : lay = _root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0)
    (hcd : c = Cfg.ofDraws lay d mk) (hord : c''.order.size = c.order.size)
    (hcr : cr.size = alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 3) + 1) {cs : Array F128} {m : Masker}
    (ha : (repState c t rep dom xs hi cr0 cr ns (alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 3) + 1)).answer c'' cs =
      .ok m) :
    _root_.Flock.Zk.rhoIndependent
      (⟨(cr.getD (alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 2)) #[])[0]!,
        (cr.getD (alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 2)) #[])[1]!⟩, ⟨cs[0]!, cs[1]!⟩) = true := by
  obtain ⟨-, sched, -, hs⟩ := repState_ok ha
  obtain ⟨hm36, hk, hn, hq⟩ := Count.sched_facts hz
  have hn0 : t.sch.n ≠ 0 := by have := (Count.sched_rho hz).1; omega
  have hm : t.st.m < 2 ^ 64 := by omega
  have hlk : lay.k0 = t.sch.k0 := by subst hlay; rfl
  have hood : lay.nOod0 = 1 := by subst hlay; rfl
  have hk' : 0 < lay.k0 := by omega
  have hsz' : c.nPiop + l0Off (t.sch.k0 + 2 + 1) = c.order.size := by
    have h1 := l0_size lay hk' hood (i := t.sch.k0 + 2) (by omega) (by omega)
    have h3 := l0Off_succ (i := t.sch.k0 + 2) (by omega)
    have hpi : c.nPiop = absPos lay (lay.lgOod 0) := by rw [hcd]; exact nPiop_ood lay d mk hk'
    have hord' : c.order.size = (_root_.Flock.Firewall.Frame.absorbed lay).size := by subst hcd; rfl
    rw [hpi, hord']
    omega
  have hpop : cr.pop.size = alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 3) := by
    rw [Array.size_pop, hcr]; omega
  obtain ⟨hsz, hL⟩ := sched_l0Run hc hal hs hm14 hdg hcap hdom hmk hx hz hlay hcd (j := t.sch.k0 + 3) le_rfl hpop
    (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[])
  rw [repState_at c t rep dom xs hi cr0 cr ns (alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 3)) hal hs (by omega)] at ha
  dsimp only at ha
  cases h2 : (_root_.Flock.Firewall.Frame.run c {} (schedReqs
      (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched cr
      (alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 3)))).2
  · rw [h2] at ha
    simp only [Bool.false_eq_true, ↓reduceIte] at ha
    rw [show alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 3) = alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 2) + 1 by
      omega, schedReqs_succ, Loop.run_append] at h2 ha
    cases hb : (_root_.Flock.Firewall.Frame.run c {} (schedReqs
        (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched cr
        (alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 2)))).2
    · simp only [hb, Bool.false_eq_true, ↓reduceIte] at h2 ha
      have hI := hL (t.sch.k0 + 2) (by omega) le_rfl hb
      rw [if_neg (show t.sch.k0 + 2 ≠ 0 by omega),
        show l0Nc (t.sch.k0 + 2) = 2 by unfold l0Nc; split_ifs <;> first | rfl | omega] at hI
      obtain ⟨-, T, sws, -, hbody⟩ := Bind.sched_l0 hal hs hdom hdg hcap hm hmk hx hk hn hq hn0 hm14
        (j := t.sch.k0 + 3) (by omega) hpop
      obtain ⟨hb1, hc1⟩ := hbody (t.sch.k0 + 3) le_rfl
      rw [FlockComplete.LigLock.l0BodyAt_r1] at hb1
      rw [show alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 3) = alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 2) + 1 by
        omega] at hb1 hc1
      rw [List.getD_eq_getElem?_getD,
        show xs.size + 2 + 1 + (t.sch.k0 + 3) = xs.size + 2 + 1 + (t.sch.k0 + 2 + 1) by omega,
        Count.openList_rho hz (c := xs.size + 2) (by omega) (e := 1) (by omega), Option.getD_some] at hc1
      have hR := rho_step hc (b := t.sch.k0 + 2) hI hsz' hb1 hc1 h2
      rcases hR with ho | ⟨ho, hl, hon, hin, hps, hrh⟩
      · exact absurd ha (answer_owed_none _ ho)
      obtain ⟨a, a', rfl⟩ := FlockComplete.JsonRT.two_of (answer_size ho ha)
      obtain ⟨g0, g1⟩ := getElem_two a a'
      rw [g0, g1]
      by_contra hd
      rw [Bool.not_eq_true] at hd
      rw [answer_rho_dependent c'' _ _ _ a a' ho hin hl (by rw [hps, hord]) hrh hd] at ha
      cases ha
    · rw [hb, if_pos rfl, hb] at h2
      cases h2
  · rw [h2] at ha
    simp only [↓reduceIte] at ha
    exact absurd ha (answer_owed_none cs rfl)

end Claims.ZKServer.RhoKeep

namespace Claims.ZKServer.LeanRelease

open Claims Claims.ZKMal
open _root_.Flock (V F128)
open Loop

set_option maxHeartbeats 800000 in
/-- **The frame refuses dependent `ρ₀, ρ₁` at `ρ₁`'s coins** (`rho_unrefused_sched`): the opening past `ρ₀`'s is the
rep's `ρ₁` round, whose coins the frame took (`coins_taken_of_live`) at the rep's state there, which takes them only
independent of `ρ₀`'s (`RhoKeep.repState_rho`); both are the opened coins (`wordsAt_slot`). -/
theorem rho_unrefused_sched_pf (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (wk : (ofLoop T L₀ real).toSrv.Worker) (x : Msg) (w : Flock.Wit) (hx : Masked.Live x) (hSv : Serves x)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (κ : (ofLoop T L₀ real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hl : ops.length ≤ refK x)
    (hlive : (ofLoop T L₀ real).toSrv.live wk x w ρ κ ((m₀, ρ₀) :: ops) = true)
    (hr : (ofLoop T L₀ real).refusesCoins x ρ ((m₀, ρ₀) :: ops) = false)
    {j r : ℕ} (hj : j < (Masked.stsOf x).size) (hr2 : r < 2) (hN : Masked.rhoSlot x j r + 1 < ops.length) :
    _root_.Flock.Zk.rhoIndependent (Masked.rhoAt (x := x) ((ZKMal.CoinLayer.session loopNr T).pad (K := refK x)
      (ops.map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2)) j r) = true := by
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨y⟩ := hx.2 j hj
  obtain ⟨-, -, hspec, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hq2 : 2 * j + r < spec.repStreams.size := by rw [FlockDeploy.repStreams_size, hspec]; omega
  have hjj : (2 * j + r) / 2 = j := by omega
  have hst : (Masked.stsOf x)[(2 * j + r) / 2]! = (Masked.stsOf x)[j] := by
    rw [hjj]
    exact getElem!_pos _ j hj
  have hstj : (Masked.stsOf x)[j]! = (Masked.stsOf x)[j] := getElem!_pos _ j hj
  have hm14 : 14 ≤ (Masked.stsOf x)[j].m := by have := (hall j hj).1; omega
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j + r) / 2]!.m = .ok (y.sch, y.tPad) := by
    rw [hst]; exact y.hz
  have hk6 : 6 ≤ (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog := by rw [hst]; exact y.wf.1.hk6
  obtain ⟨N, hNd⟩ : ∃ N, N = FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * j + r) / 2]! +
      (y.sch.k0 + 7 + 11 * (y.sch.levels.size - 1)) := ⟨_, rfl⟩
  have hrR : ∀ q, q / 2 = (2 * j + r) / 2 → FlockDeploy.repRounds true spec (Masked.stsOf x)[q / 2]! = N + 4 := by
    intro q hq'
    rw [hq']
    have e1 : (Masked.stsOf x)[(2 * j + r) / 2]!.m + (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog - 6 =
        (Masked.stsOf x)[(2 * j + r) / 2]!.m + ((Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog - 6) := by omega
    rw [repRounds_true]
    simp only [hz]
    rw [hNd]
    unfold FlockComplete.piopRoundsZk
    rw [e1]
  have hrq : ∀ q, q / 2 = (2 * j + r) / 2 → q < spec.repStreams.size →
      (roundsOf spec (Masked.stsOf x)).getD q 0 = N + 4 := by
    intro q hq' hq
    simp only [roundsOf, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hq, Option.map_some,
      Option.getD_some]
    exact hrR q hq'
  have hrsl : (roundsOf spec (Masked.stsOf x)).length = spec.repStreams.size := by simp [roundsOf]
  have hs1 : 1 ≤ FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) := by
    unfold FlockComplete.repStartZk; omega
  obtain ⟨P', hP'⟩ : ∃ P', P' = FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * j + r) / 2]! + (y.sch.k0 + 3) :=
    ⟨_, rfl⟩
  obtain ⟨s, hs⟩ : ∃ s, s = FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) + P' := ⟨_, rfl⟩
  have hbs : Masked.rhoSlot x j r + 1 = s := by
    unfold Masked.rhoSlot
    rw [Masked.repSlot_eq hsx hz]
    omega
  rw [hbs] at hN
  have hlast : (ofLoop T L₀ real).last x = refK x + 1 := loop_last loopNr T (withFrames T L₀ real) x
  obtain ⟨-, m, ha⟩ := coins_taken_of_live (ofLoop T L₀ real) wk x w ρ κ ((m₀, ρ₀) :: ops) hlive
    (by rw [hlast, List.length_cons]; omega) hr (i := s + 2) (by omega) (by simp only [List.length_cons]; omega)
  obtain ⟨⟨dm, da⟩, hd⟩ : ∃ d, ops[s]? = some d := ⟨ops[s]'(by omega), List.getElem?_eq_getElem _⟩
  have htk : ops.take (s + 1) = ops.take s ++ [(dm, da)] := by rw [List.take_add_one, hd]; rfl
  have hh : ((m₀, ρ₀) :: ops).take (s + 2) = (m₀, ρ₀) :: (ops.take s ++ [(dm, da)]) := by
    rw [List.take_succ_cons, htk]
  have hts : (ops.take s).length = s := by rw [List.length_take]; omega
  rw [hh] at ha
  change (progOf x ((ops.take s ++ [(dm, da)]).map fun p => ZKMal.CoinLayer.rd p.2)).answer
    ((ofLoop T L₀ real).cfgAt x ρ (s + 1))
    (coinsAt x ((m₀, ρ₀) :: (ops.take s ++ [(dm, da)]))) = .ok m at ha
  have hca : coinsAt x ((m₀, ρ₀) :: (ops.take s ++ [(dm, da)])) =
      (ZKMal.CoinLayer.rd da).vals (min (loopNr x s) _root_.Flock.Firewall.Frames.COIN_MAX) := by
    unfold coinsAt
    simp only [List.tail_cons, List.getLast?_append, List.getLast?_singleton, List.length_append,
      hts, List.length_singleton, Nat.add_sub_cancel]
    exact feedOf_rd _ _
  rw [hca, List.map_append, List.map_cons, List.map_nil] at ha
  have hloc : ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (s - 1) = (2 * j + r, P') := by
    have e := locate_take_sum (roundsOf spec (Masked.stsOf x)) 0 (2 * j + r) P' (by rw [hrsl]; exact hq2)
      (by rw [hrq _ rfl hq2]; omega)
    rw [Nat.zero_add] at e
    rw [← e, hs, repStartZk_eq spec _ hq2.le]
    congr 1
    omega
  have hlay : ((ofLoop T L₀ real).cfgAt x ρ (s + 1)).lay =
      ((startOf x (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (s - 1)).1).toOption.map
        (·.1)).getD default := by
    show ((startOf x (repOf x (s + 1))).toOption.map (·.1)).getD default = _
    rw [show s + 1 = s - 1 + 2 by omega, repOf_add_two hsx]
  rw [progOf_snoc x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2) (ZKMal.CoinLayer.rd da) hsx (q := s - 1)
    (by rw [List.length_map, hts]; omega), hloc] at ha
  rw [hloc] at hlay
  dsimp only at ha
  rcases hph : placeholder spec (Masked.stsOf x) (2 * j + r)
      (wordsAt x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2) 0) with e' | ⟨c', t, dom, xs, hi⟩
  · rw [hph] at ha
    exact absurd ha (answer_owed_none _ rfl)
  rw [hph] at ha
  dsimp only at ha
  obtain ⟨CR0, CR, -, hCR, ha⟩ := repState_args ha
  obtain ⟨alg, -, hal, -⟩ := repState_ok ha
  obtain ⟨hts', ⟨tp, htp⟩, hxs⟩ := Count.placeholder_tab hph
  rw [hz, Except.ok.injEq, Prod.mk.injEq] at htp
  obtain ⟨htsch, -⟩ := htp
  have hzt : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j + r) / 2]!.m = .ok (t.sch, y.tPad) := htsch ▸ hz
  obtain ⟨hm36, -, -, -⟩ := Count.sched_facts hzt
  obtain ⟨⟨hmp, hmp0, hpc⟩, hdom', hxr, -⟩ := sizesOk_at (sizesOk_of_serves hSv s0 spec _ hsx) hq2
  obtain ⟨lay, mask, hds, hc', hdm, -⟩ := placeholder_ok hph
  have hls : (wordsAt x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2) 0).size =
      min (loopNr x 0) _root_.Flock.Firewall.Frames.COIN_MAX := by
    obtain ⟨p0, rest, hpr⟩ : ∃ p0 rest, ops.take s = p0 :: rest := by
      cases h' : ops.take s with
      | nil => rw [h', List.length_nil] at hts; omega
      | cons p0 rest => exact ⟨p0, rest, rfl⟩
    rw [hpr]
    simp [wordsAt, FlockDeploy.Coin.vals]
  have hl0 : loopNr x 0 = spec.points * spec.mPts := by
    show (reqOf x 0).2 = _
    rw [reqOf_eq hsx]
    rfl
  have hpts : (wordsAt x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2) 0).size /
      (Masked.stsOf x)[(2 * j + r) / 2]!.mPts = spec.points := by
    rw [hls, hl0, Nat.min_eq_left hpc, hmp, Nat.mul_div_cancel _ hmp0]
  have hxn : xs.size = spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size := by
    rw [hxs, Count.extras_size, Count.linkPoints_size, hpts, Nat.mul_comm]
  have hxb : xs.size + 2 < 2 ^ 64 := by
    rw [hxs, Loop.extras_size]
    have := Nat.mul_le_mul_left (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size
      ((linkPoints_size_le (Masked.stsOf x)[(2 * j + r) / 2]! _).trans (hls ▸ Nat.min_le_right _ _))
    omega
  have hdg : t.st.digest.size < 2 ^ 64 := by
    rw [hts']
    simp only [_root_.Flock.Setup.ofCircuit, ByteArray.size_extract]
    omega
  have hmb : t.st.m < 2 ^ 64 := by rw [hts']; exact Nat.lt_trans hm36 (by decide)
  have hA : alg.size = ((_root_.Flock.Firewall.Session.replayCoins (Masked.stsOf x)[(2 * j + r) / 2]!.m
      (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog).toList.filter (· != 0)).length := by
    have eA := congrArg List.length (Count.rounds_counts hmb hal)
    rw [Count.piop_counts, hts', List.length_map, Array.length_toList] at eA
    exact eA
  have hplan := congrArg List.length (Count.repPlan_list (spec := spec) hzt)
  rw [Array.length_toList, repPlan_size] at hplan
  simp only [repRounds_true, hz, List.length_append] at hplan
  have h4 : ([1, _root_.Flock.Firewall.Session.INNER_CONS, 1, _root_.Flock.Zk.Q].filter (· != 0)).length = 4 := by
    decide
  have hFO : alg.size + (Count.openList (2 + spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size)
      t.sch).length = N := by
    rw [h4] at hplan
    have e1 : (Masked.stsOf x)[(2 * j + r) / 2]!.m + (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog - 6 =
        (Masked.stsOf x)[(2 * j + r) / 2]!.m + ((Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog - 6) := by omega
    rw [hA, hNd]
    unfold FlockComplete.piopRoundsZk
    rw [e1]
    exact (Nat.add_right_cancel hplan).symm
  have hx2 : xs.size + 2 = 2 + spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size := by omega
  have hkk : y.sch.k0 = t.sch.k0 := by rw [htsch]
  have hll : y.sch.levels.size = t.sch.levels.size := by rw [htsch]
  have hOL := Count.openList_length hzt
    (c := (2 + spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size)) (by omega)
  have hAP : alg.size + (2 + spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size) + 1 =
      FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * j + r) / 2]! := by
    rw [hOL] at hFO
    omega
  have hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by
    rw [hts']
    show (Masked.stsOf x)[(2 * j + r) / 2]!.merkle = _
    rw [hst]
    exact (hall j hj).2.2
  have htm : t.st.m = (Masked.stsOf x)[j]!.m := by rw [hts', hst, hstj]; rfl
  have hm14t : 14 ≤ t.st.m := by rw [htm, hstj]; exact hm14
  have hzT : _root_.Flock.Zk.schedule t.st.m = .ok (t.sch, y.tPad) := by rw [hts']; exact hzt
  obtain ⟨sch', tp', hz', hlay0, -⟩ := start_eq hds
  rw [hz, Except.ok.injEq, Prod.mk.injEq] at hz'
  have hlayT : lay = _root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0 := by
    rw [hlay0, hts', ← hz'.1, htsch]
    all_goals rfl
  have hcc : ∀ p, p < c'.order.size → c'.order[p]! < c'.h.size := by rw [hc']; exact ofDraws_order _ _ _
  have hlay' : ((ofLoop T L₀ real).cfgAt x ρ (s + 1)).lay = lay := by
    rw [hlay]
    unfold startOf
    rw [hsx]
    dsimp only
    rw [hds]
    all_goals rfl
  have hord : ((ofLoop T L₀ real).cfgAt x ρ (s + 1)).order.size = c'.order.size := by
    rw [hc']
    show (_root_.Flock.Firewall.Frame.absorbed ((ofLoop T L₀ real).cfgAt x ρ (s + 1)).lay).size = _
    rw [hlay']
    all_goals rfl
  have hCRs : CR.size = alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 3) + 1 := by
    rw [hCR, Array.size_push, Array.size_map, Array.size_range]
    omega
  rw [show P' + 1 = alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 3) + 1 by omega] at ha
  have hR := RhoKeep.repState_rho hcc hal hm14t hdg (placeholder_cap hph) (hdm ▸ hdom') hmk hxb hzT hlayT hc'
    hord hCRs ha
  have hW' : CR.getD (alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 2)) #[] =
      wordsAt x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2) (s - 1) := by
    rw [hCR, crAt_getD _ _ _ (show alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 2) < P' by omega)]
    rw [show s - 1 - P' + (alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 2)) + 1 = s - 1 by omega]
  have hplv : ∀ e < 2, loopNr x (FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) +
      (alg.size + ((2 + spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size) + 1 +
        (t.sch.k0 + 2 + e)))) = 2 := by
    intro e he
    rw [loopNr_rep hsx hq2 (by rw [hrq _ rfl hq2]; omega), hA]
    exact Count.repPlan_rho hzt he
  have hl2 : ∀ e < 2, loopNr x (s - 1 + e) = 2 := by
    intro e he
    rw [show s - 1 + e = FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) + (alg.size +
      ((2 + spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size) + 1 + (t.sch.k0 + 2 + e))) by omega]
    exact hplv e he
  have hsz0 : ∀ e < 2, e < (wordsAt x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2) (s - 1)).size := by
    intro e he
    have h0 := hl2 0 (by decide)
    rw [Nat.add_zero] at h0
    rw [wordsAt_size x ops (s := s) (p := s - 1) (by omega) (by omega), h0]
    exact lt_of_lt_of_le he (by decide)
  have hsz1 : ∀ e < 2, e < (wordsAt x ((ops.take (s + 1)).map fun p => ZKMal.CoinLayer.rd p.2) s).size := by
    intro e he
    have h1 := hl2 1 (by decide)
    rw [show s - 1 + 1 = s by omega] at h1
    rw [wordsAt_size x ops (s := s + 1) (p := s) (by omega) (by omega), h1]
    exact lt_of_lt_of_le he (by decide)
  have hw1 : wordsAt x ((ops.take (s + 1)).map fun p => ZKMal.CoinLayer.rd p.2) s =
      (ZKMal.CoinLayer.rd da).vals (min (loopNr x s) _root_.Flock.Firewall.Frames.COIN_MAX) := by
    unfold wordsAt
    rw [List.getElem?_map, List.getElem?_take_of_lt (Nat.lt_succ_self s), hd]
    rfl
  unfold Masked.rhoAt
  rw [hbs, show Masked.rhoSlot x j r = s - 1 by omega,
    ← wordsAt_slot T x ops hl (s := s) (p := s - 1) (e := 0) (by omega) (by omega) (hsz0 0 (by decide)),
    ← wordsAt_slot T x ops hl (s := s) (p := s - 1) (e := 1) (by omega) (by omega) (hsz0 1 (by decide)),
    ← wordsAt_slot T x ops hl (s := s + 1) (p := s) (e := 0) (by omega) (by omega) (hsz1 0 (by decide)),
    ← wordsAt_slot T x ops hl (s := s + 1) (p := s) (e := 1) (by omega) (by omega) (hsz1 1 (by decide)),
    hw1, ← hW']
  exact hR

end Claims.ZKServer.LeanRelease
