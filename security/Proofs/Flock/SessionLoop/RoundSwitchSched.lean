module

public import Proofs.Flock.SessionLoop.PcsRank
public import Proofs.Flock.SessionLoop.PiopState
public import Proofs.Flock.SessionLoop.RoundSwitchFrame

/-!
# A rep's schedule and frame at its ring switch's rounds

A rep's ring switch (`Drive.openStep`'s first rounds) follows its algebraic phase: round `0` carries the opening's
label, at which the frame runs the rep's rank check on the algebraic phase's coins, and the frame is past the masked
phase from then on.

* **`Bind.openStep_switch_sv`**, **`Bind.sched_switch_sv`**: as `Bind.openStep_switch_at` and `Bind.sched_switch_at`,
  with the switched vectors named: each claim's, of the zerocheck's and lincheck's claims at the points of the coins the
  schedule was fed (`FlockComplete.LigLock.Fed.PtsAt`) and then the extra claims.
* **`Ranked.repState_pre_pcs`**: where the loop's state at a rep's last algebraic round takes the round's coins, the
  frame is in the algebraic phase at its last position and holds the algebraic phase's coins, at their counts, after the
  earlier rep's claim points.
* **`Ranked.rank_of_rankFails`**: there, where the rank stop passes, the rank check holds.
* **`Ranked.repState_switched`**: where the loop's state at a later ring-switch round takes its coins, the frame is
  outside every masked phase.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer

namespace Bind

open _root_.Flock (F128 Transcript TMW)
open _root_.Flock.Firewall.Drive (Step OpenM openStep)
open _root_.Flock.Firewall.Finish (repOpening commitFw claimsOf ringSwitchP_eq switchVec)
open FlockSoundness.Refine (Frame encs)
open FlockComplete (PS asksCut ringSwitchCut)
open FlockComplete.Replay (bind_go liftTM_ok)
open FlockComplete.LigLock.Fed (feedR sentQ feed_eq CoinAt repOpening_feed cutAsks cutAsks_toC openStep_sent)
open Switch (rsBody)
open Loop

/-- **The opening's first rounds are the ring switch's, at its claims' switched vectors**: as `openStep_switch_at`,
where each claim's switched vector is the claim's in the PIOP's claims, the zerocheck's and lincheck's at the points of
the coins fed (`repPiop_rec`) and then the extra claims. -/
theorem openStep_switch_sv (t : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ) (dom : ByteArray)
    (xs : Array (Array F128)) (hi : Array ByteArray) (coins : Array (Array F128)) (hdom : dom.size < 2 ^ 64)
    (hdg : t.st.digest.size < 2 ^ 64) (hcn : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64)
    (hm : t.st.m < 2 ^ 64) (hm14 : 14 ≤ t.st.m) (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512)
    (cs : List (Array F128)) {a : ℕ} (haN : a ≤ xs.size + 2)
    (hlen : a < (sentQ (feedR (openStep t rep dom xs hi coins) cs)).length) :
    ∃ zc lc, FlockComplete.LigLock.Fed.PtsAt t.st.m t.st.fold.kLog (FlockComplete.LigLock.Fed.recCoin coins) 0 zc lc ∧
      ((sentQ (feedR (openStep t rep dom xs hi coins) cs)).take (a + 1)).map Claims.toMsg =
        (List.range (a + 1)).map (rsBody (fun k => switchVec t.g
          (_root_.Flock.abcClaims t.st.m t.st.kLog zc lc ++ claimsOf xs)[k]!) (xs.size + 2) []) := by
  rcases openStep_sent t rep dom xs hi coins hdom hdg hcn hm cs with h | ⟨r, hp, hr, hQ⟩
  · rw [h] at hlen
    simp at hlen
  have hna : r.1.1.2.2.size = xs.size + 2 := by
    have := Count.piop_na coins t.st (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0)
      (_root_.Flock.Firewall.Finish.flat t.cm0.cap) ByteArray.empty (claimsOf xs) (fun _ => 0) hdg hcn (by decide) hm _
      (Sched.trAlike_init dom hdom) #[]
    rw [hp] at this
    have h' : r.1.1.2.2.size = (claimsOf xs).size + 2 := this
    simpa only [claimsOf, Array.size_map] using h'
  have hpe : r.1.2.pending = ByteArray.empty := Switch.endE_repPiop _ _ _ _ _ _ _ _ _ _ hp
  obtain ⟨zc, lc, hcl, hpts⟩ := FlockComplete.LigLock.Fed.repPiop_rec _ _ _ _ _ _ _ hm14 _ hp
  refine ⟨zc, lc, hpts, ?_⟩
  rw [← hcl]
  obtain ⟨-, hF⟩ := repOpening_feed (fun j i => (cs.getD j #[])[i]!) a coins t.st hmk t.sch
    (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0) t.tPad rep t.g t.cm0 r.1.1.2.2
    (Array.replicate (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0).len 0) t.extra t.lanePad
    (fun l => _root_.Flock.Firewall.Finish.saltAt (hi.getD (l - 1) ByteArray.empty)) r.1.2 r.2 #[] cs (by omega)
    (coinAt_getD cs)
  rw [hQ] at hlen ⊢
  have e := hF (by simp) (by rw [Array.size_empty, Nat.zero_add]; exact hlen)
  obtain ⟨L', hL, hL'⟩ : ∃ L', ((ringSwitchCut (asksCut (fun j i => (cs.getD j #[])[i]!) a)
      (fun k => switchVec t.g r.1.1.2.2[k]!) r.1.1.2.2).run r.1.2).go #[] = (.error "past the openings", L') ∧
      L'.toList = (#[] : Array Claims.Msg).toList ++ (List.range (a + 1)).map
        (rsBody (fun k => switchVec t.g r.1.1.2.2[k]!) r.1.1.2.2.size r.1.2.pending.data.toList) := by
    rcases Nat.lt_or_ge a r.1.1.2.2.size with ha | ha
    · exact ringSwitchCut_at _ a _ r.1.1.2.2 r.1.2 #[] ha (by rw [Array.size_empty, Nat.zero_add])
    · obtain ⟨L', h1, h2⟩ := ringSwitchCut_gamma (fun j i => (cs.getD j #[])[i]!) a
        (fun k => switchVec t.g r.1.1.2.2[k]!) r.1.1.2.2 (by omega) r.1.2 #[]
        (by rw [Array.size_empty, Nat.zero_add]; omega)
      refine ⟨L', h1, ?_⟩
      rw [h2, show r.1.1.2.2.size = a by omega]
  have hro : ((repOpening (cutAsks (fun j i => (cs.getD j #[])[i]!) a)
      (fun lv g s => pure (commitFw lv g s)) t.st t.sch (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0)
      t.tPad rep t.g t.cm0 r.1.1.2.2
      (Array.replicate (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0).len 0) t.extra t.lanePad
      (fun l => _root_.Flock.Firewall.Finish.saltAt (hi.getD (l - 1) ByteArray.empty))).run r.1.2).go #[] =
      (.error "past the openings", L') := by
    unfold repOpening
    rw [ringSwitchP_eq, cutAsks_toC]
    exact FlockComplete.bind_go_err hL
  have e2 := congrArg Array.toList (Prod.mk.inj (hro.symm.trans e)).2
  rw [hL', hpe, empty_toList, hna] at e2
  simp only [Array.toList_append, List.nil_append, Array.size_empty, Nat.sub_zero] at e2
  exact e2.symm

/-- **A rep's schedule fed `a` of the opening's coins, at its claims' switched vectors**: as `sched_switch_at`, where
each claim's switched vector is the claim's in the PIOP's claims at the points of the algebraic phase's coins
(`openStep_switch_sv`). -/
theorem sched_switch_sv {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} {cr : Array (Array F128)} {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr = .ok sched) (hdom : dom.size < 2 ^ 64)
    (hdg : t.st.digest.size < 2 ^ 64) (hcn : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64)
    (hm : t.st.m < 2 ^ 64) (hm14 : 14 ≤ t.st.m) (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512)
    (hx : xs.size + 2 < 2 ^ 64) (hk : 1 ≤ t.sch.k0) (hn : t.sch.n < 2 ^ 62)
    (hq : ∀ i, (t.sch.levels.getD i default).q < 2 ^ 63) {a : ℕ} (hcr : cr.size = alg.size + a)
    (haN : a ≤ xs.size + 2) :
    sched.size = alg.size + a + 1 ∧ ∃ zc lc,
      FlockComplete.LigLock.Fed.PtsAt t.st.m t.st.fold.kLog
        (FlockComplete.LigLock.Fed.recCoin (cr.extract 0 alg.size)) 0 zc lc ∧
      ∀ k ≤ a, (sched.getD (alg.size + k) (.empty, 0)).1.data.toList = rsBody (fun i => switchVec t.g
          (_root_.Flock.abcClaims t.st.m t.st.kLog zc lc ++ claimsOf xs)[i]!) (xs.size + 2) [] k ∧
        (sched.getD (alg.size + k) (.empty, 0)).2 = (Count.openList (xs.size + 2) t.sch).getD k 0 := by
  obtain ⟨hdrop, hnf⟩ := schedSoFar_sentQ hal hs (by omega)
  have hcsl : (cr.extract alg.size cr.size).toList.length = a := by
    simp only [Array.length_toList, Array.size_extract]
    omega
  have hlenF := feed_part _ _ _
    (Count.openStep_end t rep dom xs hi (cr.extract 0 alg.size) hdom hdg hcn hm hx hk hn hq)
    (by rw [hcsl]; exact lt_of_le_of_lt haN (openList_len t.sch (by omega))) hnf
  rw [hcsl] at hlenF
  have hsz : sched.size = alg.size + a + 1 := by
    have := congrArg List.length hdrop
    simp only [List.length_drop, List.length_map, Array.length_toList] at this
    omega
  obtain ⟨zc, lc, hpts, hbody⟩ := openStep_switch_sv t rep dom xs hi (cr.extract 0 alg.size) hdom hdg hcn hm hm14
    hmk (cr.extract alg.size cr.size).toList haN (by omega)
  exact ⟨hsz, zc, lc, hpts, fun k hk' => ⟨sched_bytes hdrop hbody (by omega) hk',
    sched_count hal hs hdom hdg hcn hm hx hk hn hq (by omega)⟩⟩

end Bind

namespace Ranked

open Claims Claims.ZKMal
open _root_.Flock (V F128)
open _root_.Flock.Firewall.Frame (Cfg Masker MaskSlot K_SKIP N_INNER LABEL_ZC maskRank claimPoints)
open FlockSoundness.Refine (Frame encs)
open Loop

/-- **The frame taking a rep's last algebraic round's coins**: where the frame holds the rep's phase and rank (`Inv`)
after round `a` and takes round `a`'s coins into a state in the algebraic phase outside level 0, it was before the
opening's label: the coins it holds are rounds `0 … a`'s, each answered the round's count, after the earlier rep's
points. -/
theorem inv_answer_pre {c c'' : Cfg} {p : Array (Array F128)} {cs : ℕ → Array F128} {nn : ℕ → ℕ} {a : ℕ}
    {S m : Masker} (hI : Inv c p cs nn a S) (ha : S.answer c'' (cs a) = .ok m) (hon : m.on = true)
    (hl : m.l0 = false) :
    m.done = false ∧ m.prior = p ∧ m.coins.toList = flatR cs (a + 1) ∧ ∀ i < a + 1, (cs i).size = nn i := by
  obtain ⟨hg, hp, hsz, ho, d, -, hco, hdk⟩ := hI
  have hn : S.owed = some (nn a) := by
    rcases ho with ho | ho
    · exact absurd ha (answer_owed_none _ ho)
    · exact ho.1
  have hsize := answer_size hn ha
  have hdS : S.done = false := by
    cases hd' : S.done
    · rfl
    · exfalso
      have hol := (hg.2 hd').1
      cases hi : S.inner with
      | some k =>
        obtain ⟨-, b, l, -, -⟩ := answer_proj (Or.inl (by rw [hi]; rfl)) ha
        have h1 := hol (by rw [← b]; exact hon)
        rw [← l, hl] at h1
        cases h1
      | none =>
        cases hl0 : S.l0
        · have e := answer_alg c'' S (cs a) hi hl0 ha
          have hon' : S.on = true := by rw [e] at hon; exact hon
          have h1 := hol hon'
          rw [hl0] at h1
          cases h1
        · obtain ⟨-, -, l, -, -⟩ := answer_proj (Or.inr hl0) ha
          rw [l, hl0] at hl
          cases hl
  obtain ⟨hon', hl', hi'⟩ := hg.1 hdS
  have e := answer_alg c'' S (cs a) hi' hl' ha
  refine ⟨by rw [e]; exact hdS, by rw [e]; exact hp, ?_, fun i hi => ?_⟩
  · rw [e]
    simp only [hon', ↓reduceIte, Array.toList_append]
    rw [hco, hdk hdS, flatR_succ]
  · rcases Nat.lt_succ_iff_lt_or_eq.1 hi with hi | rfl
    · exact (hsz i hi).1
    · exact hsize

/-- **At a rep's last algebraic round the frame holds the algebraic phase's coins**: where the loop's state there
(`m + k_log − 6` rounds in) takes the round's coins `cs`, the frame is in the algebraic phase, outside level 0, at its
absorbed position after the phase (`repState_piop`), holding rounds `0 … m + k_log − 7`'s coins (`cr`), answered at
the algebraic phase's counts, after the earlier rep's claim points (`cr0`'s rounds, sent as bytes) at rep `1`
(`run_head`, `inv_answer_pre`). -/
theorem repState_pre_pcs {c c'' : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size)
    {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} {cr0 cr : Array (Array F128)} {ns : ℕ → ℕ} {alg : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hm14 : 14 ≤ t.st.m) (hk6 : 6 ≤ t.st.fold.kLog) (hm : t.st.m < 2 ^ 64) (hdg : t.st.digest.size < 2 ^ 64)
    (hcap : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64) (hdom : dom.size < 2 ^ 64)
    (hb : (piopV t.st.m t.st.fold.kLog t.st.digest (_root_.Flock.Firewall.Finish.flat t.cm0.cap) Sched.root0
      (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0) (fun _ => 0) alg.size).length ≤ c.order.size)
    {cs : Array F128} (hcs : cs = cr.getD (alg.size - 1) #[]) {m : Masker}
    (ha : (repState c t rep dom xs hi cr0 cr ns alg.size).answer c'' cs = .ok m) :
    alg.size = t.st.m + t.st.fold.kLog - 6 ∧ (∀ k < alg.size, (cr.getD k #[]).size = cnt t.st.m k) ∧
      m.on = true ∧ m.done = false ∧ m.l0 = false ∧
      m.pos = (piopV t.st.m t.st.fold.kLog t.st.digest (_root_.Flock.Firewall.Finish.flat t.cm0.cap) Sched.root0
        (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0) (fun _ => 0) alg.size).length ∧
      m.coins.toList = flatR (fun k => cr.getD k #[]) alg.size ∧
      (rep = 0 → m.prior = #[]) ∧
      (rep ≠ 0 → ((cr0.extract 0 alg.size).foldl (· ++ ·) #[]).isEmpty = false →
        claimPoints c.lay (_root_.Flock.Firewall.Frame.words
          (_root_.Flock.Firewall.Frame.wordBytes ((cr0.extract 0 alg.size).foldl (· ++ ·) #[])) 0
          ((cr0.extract 0 alg.size).foldl (· ++ ·) #[]).size) = .ok m.prior) := by
  obtain ⟨-, sched, -, hs⟩ := repState_ok ha
  obtain ⟨rest, hrest⟩ := schedSoFar_pre hal hs
  have hg : ∀ k, k < alg.size → sched.getD k (.empty, 0) = (alg[k]?).getD (.empty, 0) := fun k hk' => by
    rw [hrest, Array.getD_eq_getD_getElem?, Array.getElem?_append_left hk']
  obtain ⟨hA, hcnt⟩ := alg_cnt hm14 hk6 hm hal
  obtain ⟨hsz, -⟩ := Sched.rounds_piop hm14 hm hal
  obtain ⟨ha0, ha1⟩ := Sched.rounds_two hm14 hm hal
  have h1s : 1 < alg.size := (Array.getElem?_eq_some_iff.1 ha1).1
  have h0 : sched.getD 0 (.empty, 0) = (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8,
      .bytes t.st.digest, .bytes (_root_.Flock.Firewall.Finish.flat t.cm0.cap),
      .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray, 6) := by
    rw [hg 0 (by omega), ha0, Option.getD_some, head_encs]
  obtain ⟨a, hA1⟩ : ∃ a, alg.size = a + 1 := ⟨alg.size - 1, by omega⟩
  have hSs : alg.size ≤ sched.size := by rw [hrest, Array.size_append]; omega
  have ha' := ha
  rw [hA1] at ha' hb
  obtain ⟨hon, -, hl, hpos⟩ := repState_piop hc hal hs hm14 hm hdg hcap hdom (a := a) (by omega) hb ha'
  rw [repState_at c t rep dom xs hi cr0 cr ns a hal hs (by omega)] at ha'
  dsimp only at ha'
  rw [hA1, Nat.add_sub_cancel] at hcs
  subst hcs
  cases h2 : (_root_.Flock.Firewall.Frame.run c {} (schedReqs
      (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched cr a)).2
  · rw [h2] at ha'
    simp only [Bool.false_eq_true, ↓reduceIte] at ha'
    have hS : 0 < sched.size := by omega
    rw [schedReqs_pairs _ sched #[] cr a (by omega)] at h2 ha'
    obtain ⟨p, hp1, hp2, he⟩ := run_prior_eq c _ _ h2
    rw [he] at h2 ha'
    have hq0 : qOf sched #[] 0 = .round 6 (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8,
        .bytes t.st.digest, .bytes (_root_.Flock.Firewall.Finish.flat t.cm0.cap),
        .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray)
        (.sched (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes t.st.digest,
          .bytes (_root_.Flock.Firewall.Finish.flat t.cm0.cap), .label LABEL_ZC.toUTF8] ++
          (Frame.sq 2 6).enc).toArray)) := by
      simp only [qOf, hS, ↓reduceIte, h0]
    rw [hq0] at h2 ha'
    have hI := run_head hc p hdg hcap hdom (qOf_round sched #[]) (by simp only [nnOf, hS, ↓reduceIte, h0])
      (show 0 < a by omega) h2
    obtain ⟨hd, hpr, hco, hcs⟩ := inv_answer_pre hI ha' hon hl
    refine ⟨hA, fun k hk' => ?_, hon, hd, hl, by rw [hA1]; exact hpos, by rw [hA1]; exact hco, fun h => ?_,
      fun h hne => ?_⟩
    · refine (hcs k (by omega)).trans ?_
      simp only [nnOf, show k < sched.size by omega, ↓reduceIte]
      rw [hg k hk']
      exact hcnt k hk'
    · rw [ite_eq_left h] at hp1
      rw [hpr]
      exact hp1 rfl
    · rw [ite_eq_right h] at hp2
      rw [hpr]
      exact hp2 hne
  · rw [h2] at ha'
    simp only [↓reduceIte] at ha'
    exact absurd ha' (answer_owed_none _ rfl)

/-- **Where the rank stop passes in the algebraic phase at its last position, the rank check holds** on the claim
points of the coins the frame holds, after the earlier rep's. -/
theorem rank_of_rankFails {c : Cfg} {s : Masker} (hf : LeanRelease.rankFails c s = false) (hon : s.on = true)
    (hd : s.done = false) (hl : s.l0 = false) (hp : s.pos = c.nPiop) {ms : MaskSlot} (hm : c.mask = some ms)
    {P : Array (Array F128)} (hP : claimPoints c.lay s.coins = .ok P) :
    maskRank (s.prior ++ P) ms (c.lay.nLc + K_SKIP) = 128 * (s.prior ++ P).size := by
  simpa [LeanRelease.rankFails, hon, hd, hl, hp, hm, hP] using hf

/-- **Past the ring switch's first round the frame is outside every masked phase**: where the loop's state at a rep's
ring-switch round `k ≥ 1`, at most the claims' count and `γ`'s, takes the round's coins, the frame took the opening's
label (`Bind.sched_switch_at`, `Flags.sched_flags`) and is past the masked phase. -/
theorem repState_switched {c c'' : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size)
    {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} {cr0 cr : Array (Array F128)} {ns : ℕ → ℕ} {alg : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hm14 : 14 ≤ t.st.m) (hm : t.st.m < 2 ^ 64) (hdg : t.st.digest.size < 2 ^ 64)
    (hcap : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64) (hdom : dom.size < 2 ^ 64)
    (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (hx : xs.size + 2 < 2 ^ 64) (hk : 1 ≤ t.sch.k0)
    (hn : t.sch.n < 2 ^ 62) (hq : ∀ i, (t.sch.levels.getD i default).q < 2 ^ 63) {k : ℕ} (hk0 : k ≠ 0)
    (hkN : k ≤ xs.size + 2) (hcr : cr.size = alg.size + k) {cs : Array F128} {m : Masker}
    (ha : (repState c t rep dom xs hi cr0 cr ns (alg.size + k)).answer c'' cs = .ok m) :
    m.on = false := by
  obtain ⟨-, sched, -, hs⟩ := repState_ok ha
  obtain ⟨rest, hrest⟩ := schedSoFar_pre hal hs
  have hg : ∀ k, k < alg.size → sched.getD k (.empty, 0) = (alg[k]?).getD (.empty, 0) := fun k hk' => by
    rw [hrest, Array.getD_eq_getD_getElem?, Array.getElem?_append_left hk']
  obtain ⟨ha0, ha1⟩ := Sched.rounds_two hm14 hm hal
  have h1s : 1 < alg.size := (Array.getElem?_eq_some_iff.1 ha1).1
  have h0 : sched.getD 0 (.empty, 0) = (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8,
      .bytes t.st.digest, .bytes (_root_.Flock.Firewall.Finish.flat t.cm0.cap),
      .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray, 6) := by
    rw [hg 0 (by omega), ha0, Option.getD_some, head_encs]
  have hpop : cr.pop.size = alg.size + (k - 1) := by rw [Array.size_pop, hcr]; omega
  obtain ⟨hsz, sv, hsv, hB⟩ := Bind.sched_switch_at hal hs hdom hdg hcap hm hmk hx hk hn hq hpop (by omega)
  have hN0 : xs.size + 2 ≠ 0 := by omega
  have hsv' : ∀ i, (sv i).size < 2 ^ 64 := fun i => by rw [hsv]; decide
  have hp : Flags.PcsRound (sched.getD alg.size (.empty, 0)) := by
    obtain ⟨hb, h7⟩ := hB 0 (Nat.zero_le _)
    rw [Bind.openList_lt t.sch hN0 (by omega)] at h7
    rw [Nat.add_zero] at hb h7
    rw [show sched.getD alg.size (.empty, 0) = ((sched.getD alg.size (.empty, 0)).1, 7) from Prod.ext rfl h7]
    exact Switch.pcsRound_rs sv hN0 (hsv' 0) hb
  have hq' : ∀ k', 1 ≤ k' → k' ≤ alg.size + (k - 1) → k' ≠ alg.size →
      Quiet.QRound (sched.getD k' (.empty, 0)) := by
    intro k' h1 hkJ hkA
    rcases Nat.lt_or_ge k' alg.size with hkl | hkl
    · rw [hg k' hkl, Array.getElem?_eq_getElem hkl, Option.getD_some]
      exact Quiet.rounds_quiet hm hdg hcap hdom hal k' _ h1 (Array.getElem?_eq_getElem hkl)
    · obtain ⟨i, rfl⟩ : ∃ i, k' = alg.size + i := ⟨k' - alg.size, by omega⟩
      obtain ⟨hb, h7⟩ := hB i (by omega)
      rw [Bind.openList_lt t.sch hN0 (by omega)] at h7
      rw [show sched.getD (alg.size + i) (.empty, 0) = ((sched.getD (alg.size + i) (.empty, 0)).1, 7) from
        Prod.ext rfl h7]
      exact Switch.qRound_rs sv (by omega) (by omega) (hsv' i) hb
  rw [show alg.size + k = alg.size + (k - 1) + 1 by omega] at ha
  rw [repState_at c t rep dom xs hi cr0 cr ns (alg.size + (k - 1)) hal hs (by omega)] at ha
  dsimp only at ha
  cases h2 : (_root_.Flock.Firewall.Frame.run c {} (schedReqs
      (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched cr (alg.size + (k - 1)))).2
  · rw [h2] at ha
    simp only [Bool.false_eq_true, ↓reduceIte] at ha
    rcases Flags.sched_flags hc _ sched cr hdg hcap hdom h0 (by omega) hp _ hq' h2 with ho | ⟨-, hF⟩
    · exact absurd ha (answer_owed_none cs ho)
    · have e := answer_alg c'' _ cs hF.2.2.1 hF.1 ha
      rw [e]
      exact (hF.2.2.2.2 (by omega)).2.1
  · rw [h2] at ha
    simp only [↓reduceIte] at ha
    exact absurd ha (answer_owed_none cs rfl)

end Ranked

end Claims.ZKServer
