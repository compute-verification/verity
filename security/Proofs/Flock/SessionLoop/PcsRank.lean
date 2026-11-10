module

public import Proofs.Flock.SessionLoop.CutPiopLive
public import Proofs.Flock.Claims.ZK.L0PinBind
public import Proofs.Flock.Claims.ZK.FrameRankFault

/-!
# A rep's rank check at its opening's label, and the ring switch's rounds so far

A rep's opening (`Drive.openStep`) begins with the ring switch: its first round carries the opening's label, at which
the frame runs the rep's rank check (`Masker.label`) on the algebraic phase's coins. This file states the schedule's
ring switch fed any number of its coins, and the rank check there:

* **`Switch.forIn_go_cut`**, **`Bind.ringSwitchCut_at`**: the cut ring switch, cut at its round `i` below the claims',
  stops having sent its first `i + 1` bodies (`rsBody`).
* **`Bind.openStep_switch_at`**, **`Bind.sched_switch_at`**: a rep's schedule fed `a` of the opening's coins, up to
  `γ`'s, holds the ring switch's first `a + 1` rounds, at the opening's counts.
* **`Ranked.schedReqs_rank`**, **`Ranked.repState_rank_pcs`**: at the loop's state at a rep's opening's first round,
  where it takes the round's coins, the frame held the algebraic phase's coins at its counts, and the rank check held on
  them after the earlier rep's points.
* **`LeanRelease.rank_unrefused_sched`**: on a live history past table `j`'s rep `r`'s rank check's round
  (`Masked.rankSlot`), the rep's rank check holds at the opened coins.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer

namespace Switch

open _root_.Flock (F128 Transcript TMW)
open FlockComplete (PS)
open FlockComplete.Replay (bind_go)

/-- **A loop cut at its step `i`**: steps before `i` yield, step `i` throws; the loop, then any continuation, throws
having sent the first `i + 1` steps' bodies. -/
theorem forIn_go_cut {σ β : Type} (body : ℕ → σ → TMW PS (ForInStep σ)) (T : ℕ → Flock.Transcript)
    (B : ℕ → Claims.Msg) (L n i : ℕ) (hi : i < n) {t₀ : Flock.Transcript} (hT : T 0 = t₀) (e : String)
    (hbody : ∀ k < i, ∀ (s : σ) (l : Array Claims.Msg), l.size = L + k →
      ∃ s', ((body k s).run (T k)).go l = (.ok (.yield s', T (k + 1)), l.push (B k)))
    (hcut : ∀ (s : σ) (l : Array Claims.Msg), l.size = L + i →
      ((body i s).run (T i)).go l = (.error e, l.push (B i)))
    (s₀ : σ) (f : σ → TMW PS β) (l : Array Claims.Msg) (hl : l.size = L) :
    ((forIn [:n] s₀ body >>= f).run t₀).go l = (.error e, l ++ ((List.range (i + 1)).map B).toArray) := by
  subst hT
  refine FlockComplete.bind_go_err ?_
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices h : ∀ k a (s : σ) (l' : Array Claims.Msg), a ≤ i → i < a + k → l'.size = L + a →
      l'.toList = l.toList ++ (List.range a).map B →
      ((forIn (List.range' a k) s body).run (T a)).go l' =
        (.error e, l ++ ((List.range (i + 1)).map B).toArray) by
    exact h n 0 s₀ l (Nat.zero_le _) (by omega) (by omega) (by simp)
  intro k
  induction k with
  | zero =>
    intro a s l' ha hk _ _
    omega
  | succ k ih =>
    intro a s l' ha hk hl' hL
    rw [List.range'_succ, List.forIn_cons]
    rcases Nat.lt_or_ge a i with hai | hai
    · obtain ⟨s', hs'⟩ := hbody a hai s l' hl'
      rw [bind_go hs']
      dsimp only
      exact ih (a + 1) s' (l'.push (B a)) (by omega) (by omega) (by simp [hl']; omega)
        (by rw [Array.toList_push, hL, List.range_succ, List.map_append, List.map_cons, List.map_nil,
          List.append_assoc])
    · obtain rfl : a = i := by omega
      rw [FlockComplete.bind_go_err (hcut s l' hl')]
      congr 1
      apply Array.toList_inj.1
      rw [Array.toList_push, hL, Array.toList_append, List.toList_toArray, List.range_succ, List.map_append,
        List.map_cons, List.map_nil, List.append_assoc]

end Switch

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

/-- **The cut ring switch, cut at a claim's round**: from `l`, cut at its round `i` below the claims', it stops having
sent `rsBody`'s first `i + 1` bodies after `t`'s pending bytes. -/
theorem ringSwitchCut_at (coin : ℕ → ℕ → F128) (n : ℕ) (sv : ℕ → Array F128)
    (claims : Array _root_.Flock.Claim) (t : Flock.Transcript) (l : Array Claims.Msg) {i : ℕ} (hi : i < claims.size)
    (hl : n = l.size + i) :
    ∃ L', ((ringSwitchCut (asksCut coin n) sv claims).run t).go l = (.error "past the openings", L') ∧
      L'.toList = l.toList ++ (List.range (i + 1)).map (rsBody sv claims.size t.pending.data.toList) := by
  unfold ringSwitchCut
  dsimp only
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.label_eq _ _) _)]
  try dsimp only
  refine ⟨_, Switch.forIn_go_cut _ (Switch.rsLoopT sv coin l.size (Switch.pcsT t))
    (Switch.rsLoopB sv coin l.size (Switch.pcsT t)) l.size claims.size i hi rfl "past the openings" ?hbody ?hcut _ _ l
    rfl, ?_⟩
  case hbody =>
    intro k hk s l' hl'
    try dsimp only
    rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.label_eq _ _) _)]
    rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbSlice_eq _ _) _)]
    rw [bind_go (sampleVecP_cut_ok coin (by decide) _ (by omega))]
    rw [hl']
    exact ⟨_, rfl⟩
  case hcut =>
    intro s l' hl'
    try dsimp only
    rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.label_eq _ _) _)]
    rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbSlice_eq _ _) _)]
    rw [FlockComplete.bind_go_err (x := FlockComplete.sampleVecP (asksCut coin n) 7)
      (FlockComplete.squeezeP_cut_err coin 2 (by decide) _ (by omega))]
    rfl
  rw [Array.toList_append, List.toList_toArray]
  congr 1
  refine List.map_congr_left fun k hk => ?_
  rw [List.mem_range] at hk
  have e := congrArg (fun L => L[k]?) (Switch.rsLoopB_map sv coin l.size t claims.size)
  simp only [List.getElem?_map, List.getElem?_range (show k < claims.size by omega), Option.map_some,
    Option.some.injEq] at e
  exact e

/-- **The opening's first rounds are the ring switch's, fed any of its coins**: fed coins for `a` rounds, up to `γ`'s,
`openStep`'s first `a + 1` rounds are the switch's bodies from empty pending bytes, each claim's switched vector of
`128` words. -/
theorem openStep_switch_at (t : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ) (dom : ByteArray)
    (xs : Array (Array F128)) (hi : Array ByteArray) (coins : Array (Array F128)) (hdom : dom.size < 2 ^ 64)
    (hdg : t.st.digest.size < 2 ^ 64) (hcn : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64)
    (hm : t.st.m < 2 ^ 64) (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (cs : List (Array F128))
    {a : ℕ} (haN : a ≤ xs.size + 2) (hlen : a < (sentQ (feedR (openStep t rep dom xs hi coins) cs)).length) :
    ∃ sv : ℕ → Array F128, (∀ k, (sv k).size = 128) ∧
      ((sentQ (feedR (openStep t rep dom xs hi coins) cs)).take (a + 1)).map Claims.toMsg =
        (List.range (a + 1)).map (rsBody sv (xs.size + 2) []) := by
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
  refine ⟨fun k => switchVec t.g r.1.1.2.2[k]!, fun k => by simp [switchVec], ?_⟩
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

/-- **A rep's schedule fed `a` of the opening's coins**, up to `γ`'s: past its algebraic phase it holds the ring
switch's first `a + 1` rounds, its bodies from empty pending bytes and the opening's counts, and nothing more. -/
theorem sched_switch_at {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} {cr : Array (Array F128)} {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr = .ok sched) (hdom : dom.size < 2 ^ 64)
    (hdg : t.st.digest.size < 2 ^ 64) (hcn : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64)
    (hm : t.st.m < 2 ^ 64) (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (hx : xs.size + 2 < 2 ^ 64)
    (hk : 1 ≤ t.sch.k0) (hn : t.sch.n < 2 ^ 62) (hq : ∀ i, (t.sch.levels.getD i default).q < 2 ^ 63) {a : ℕ}
    (hcr : cr.size = alg.size + a) (haN : a ≤ xs.size + 2) :
    sched.size = alg.size + a + 1 ∧ ∃ sv : ℕ → Array F128, (∀ k, (sv k).size = 128) ∧ ∀ k ≤ a,
      (sched.getD (alg.size + k) (.empty, 0)).1.data.toList = rsBody sv (xs.size + 2) [] k ∧
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
  obtain ⟨sv, hsv, hbody⟩ := openStep_switch_at t rep dom xs hi (cr.extract 0 alg.size) hdom hdg hcn hm hmk
    (cr.extract alg.size cr.size).toList haN (by omega)
  exact ⟨hsz, sv, hsv, fun k hk' => ⟨sched_bytes hdrop hbody (by omega) hk',
    sched_count hal hs hdom hdg hcn hm hx hk hn hq (by omega)⟩⟩

end Bind

namespace Ranked

open Claims Claims.ZKMal
open _root_.Flock (V F128)
open _root_.Flock.Firewall.Frame (Cfg Masker MaskSlot K_SKIP N_INNER LABEL_ZC maskRank claimPoints)
open FlockSoundness.Refine (Frame encs)
open Loop

/-- **At a rep's scheduled round `J` past the opening's label the rank check held**: where a rep's requests up to it
(`schedReqs`) run on with the frame past the label, the frame holds rounds `0 … d − 1`'s coins, each answered the
round's count, and the rank check held there after the earlier rep's claim points. -/
theorem schedReqs_rank {c : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) (P : Array F128)
    (sched : Array (ByteArray × ℕ)) (cr : Array (Array F128)) {digest cap dom : ByteArray}
    (hdg : digest.size < 2 ^ 64) (hcap : cap.size < 2 ^ 64) (hdom : dom.size < 2 ^ 64)
    (h0 : sched.getD 0 (.empty, 0) = (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes digest,
      .bytes cap, .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray, 6)) {J : ℕ} (hJ0 : 0 < J)
    (hJ : J < sched.size)
    (h2 : (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr J)).2 = false)
    (hdone : (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr J)).1.m.done = true) :
    ∃ p, (P.isEmpty = true → p = #[]) ∧
      (P.isEmpty = false → claimPoints c.lay
        (_root_.Flock.Firewall.Frame.words (_root_.Flock.Firewall.Frame.wordBytes P) 0 P.size) = .ok p) ∧
      RankHeld c (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr J)).1.m ∧
      (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr J)).1.m.prior = p ∧
      ∃ d ≤ J, (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr J)).1.m.coins.toList =
          flatR (fun k => cr.getD k #[]) d ∧
        ∀ k < J, (cr.getD k #[]).size = (sched.getD k (.empty, 0)).2 ∧ 1 ≤ (sched.getD k (.empty, 0)).2 := by
  have hS : 0 < sched.size := by omega
  rw [schedReqs_pairs P sched #[] cr J hJ] at h2 hdone ⊢
  obtain ⟨p, hp1, hp2, he⟩ := run_prior_eq c P _ h2
  rw [he] at h2 hdone ⊢
  have hq0 : qOf sched #[] 0 = .round 6 (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8,
      .bytes digest, .bytes cap, .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray)
      (.sched (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap,
        .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray)) := by
    simp only [qOf, hS, ↓reduceIte, h0]
  rw [hq0] at h2 hdone ⊢
  obtain ⟨hg, hpr, hsz, -, d, hd, hco, -⟩ := run_head hc p hdg hcap hdom (qOf_round sched #[])
    (by simp only [nnOf, hS, ↓reduceIte, h0]) hJ0 h2
  refine ⟨p, hp1, hp2, (hg.2 hdone).2, hpr, d, hd, hco, fun k hk => ?_⟩
  have e := hsz k hk
  simp only [nnOf, show k < sched.size by omega, ↓reduceIte] at e
  exact e

/-- **At a rep's opening's first round the rank check held on the algebraic phase's coins**: where the loop's state
there takes the round's coins, the frame held the rounds `0 … m + k_log − 7`'s coins (`cr`), answered at the algebraic
phase's counts, and the rank check held at their claim points, after the earlier rep's (`cr0`'s rounds, sent as bytes)
at rep `1`. The round is the ring switch's first (`Bind.sched_switch_at`), the opening's label (`Flags.PcsRound`), which
the frame takes past its quiet rounds (`Flags.sched_flags`). -/
theorem repState_rank_pcs {c c'' : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size)
    {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} {cr0 cr : Array (Array F128)} {ns : ℕ → ℕ} {alg : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hm14 : 14 ≤ t.st.m) (hk6 : 6 ≤ t.st.fold.kLog) (hm : t.st.m < 2 ^ 64) (hdg : t.st.digest.size < 2 ^ 64)
    (hcap : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64) (hdom : dom.size < 2 ^ 64)
    (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (hx : xs.size + 2 < 2 ^ 64) (hk : 1 ≤ t.sch.k0)
    (hn : t.sch.n < 2 ^ 62) (hq : ∀ i, (t.sch.levels.getD i default).q < 2 ^ 63)
    (hcr : cr.size = alg.size + 1) {ms : MaskSlot} (hmask : c.mask = some ms) {k0 : ℕ}
    (hlay : c.lay = _root_.Flock.Zk.PadLayout.new t.st.m t.st.fold.kLog k0) {cs : Array F128} {m : Masker}
    (ha : (repState c t rep dom xs hi cr0 cr ns (alg.size + 1)).answer c'' cs = .ok m) :
    alg.size = t.st.m + t.st.fold.kLog - 6 ∧ (∀ k < alg.size, (cr.getD k #[]).size = cnt t.st.m k) ∧
      ∃ S : Masker, ∃ P, S.coins.toList = flatR (fun k => cr.getD k #[]) alg.size ∧ claimPoints c.lay S.coins = .ok P ∧
        maskRank (S.prior ++ P) ms (c.lay.nLc + K_SKIP) = 128 * (S.prior ++ P).size ∧
        (rep = 0 → S.prior = #[]) ∧
        (rep ≠ 0 → ((cr0.extract 0 alg.size).foldl (· ++ ·) #[]).isEmpty = false →
          claimPoints c.lay (_root_.Flock.Firewall.Frame.words
            (_root_.Flock.Firewall.Frame.wordBytes ((cr0.extract 0 alg.size).foldl (· ++ ·) #[])) 0
            ((cr0.extract 0 alg.size).foldl (· ++ ·) #[]).size) = .ok S.prior) := by
  obtain ⟨-, sched, -, hs⟩ := repState_ok ha
  obtain ⟨rest, hrest⟩ := schedSoFar_pre hal hs
  have hg : ∀ k, k < alg.size → sched.getD k (.empty, 0) = (alg[k]?).getD (.empty, 0) := fun k hk' => by
    rw [hrest, Array.getD_eq_getD_getElem?, Array.getElem?_append_left hk']
  obtain ⟨hA, hcnt⟩ := alg_cnt hm14 hk6 hm hal
  obtain ⟨ha0, ha1⟩ := Sched.rounds_two hm14 hm hal
  have h1s : 1 < alg.size := (Array.getElem?_eq_some_iff.1 ha1).1
  have h0 : sched.getD 0 (.empty, 0) = (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8,
      .bytes t.st.digest, .bytes (_root_.Flock.Firewall.Finish.flat t.cm0.cap),
      .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray, 6) := by
    rw [hg 0 (by omega), ha0, Option.getD_some, head_encs]
  have hpop : cr.pop.size = alg.size + 0 := by rw [Array.size_pop, hcr]; omega
  obtain ⟨hsz, sv, hsv, hB⟩ := Bind.sched_switch_at hal hs hdom hdg hcap hm hmk hx hk hn hq hpop (Nat.zero_le _)
  have hN0 : xs.size + 2 ≠ 0 := by omega
  have hp : Flags.PcsRound (sched.getD alg.size (.empty, 0)) := by
    obtain ⟨hb, h7⟩ := hB 0 le_rfl
    rw [Bind.openList_lt t.sch hN0 (by omega)] at h7
    rw [Nat.add_zero] at hb h7
    rw [show sched.getD alg.size (.empty, 0) = ((sched.getD alg.size (.empty, 0)).1, 7) from Prod.ext rfl h7]
    exact Switch.pcsRound_rs sv hN0 (by rw [hsv]; decide) hb
  have hq' : ∀ k, 1 ≤ k → k ≤ alg.size → k ≠ alg.size → Quiet.QRound (sched.getD k (.empty, 0)) := by
    intro k h1 hkJ hkA
    have hkl : k < alg.size := by omega
    rw [hg k hkl, Array.getElem?_eq_getElem hkl, Option.getD_some]
    exact Quiet.rounds_quiet hm hdg hcap hdom hal k _ h1 (Array.getElem?_eq_getElem hkl)
  rw [repState_at c t rep dom xs hi cr0 cr ns alg.size hal hs (by omega)] at ha
  dsimp only at ha
  cases h2 : (_root_.Flock.Firewall.Frame.run c {} (schedReqs
      (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched cr alg.size)).2
  · rw [h2] at ha
    simp only [Bool.false_eq_true, ↓reduceIte] at ha
    have hdone : (_root_.Flock.Firewall.Frame.run c {} (schedReqs
        (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched cr alg.size)).1.m.done =
        true := by
      rcases Flags.sched_flags hc _ sched cr hdg hcap hdom h0 (by omega) hp _ hq' h2 with ho | ⟨-, hF⟩
      · exact absurd ha (answer_owed_none cs ho)
      · exact (hF.2.2.2.2 le_rfl).1
    obtain ⟨p, hp1, hp2, hRk, hpr, d, hd, hco, hsz'⟩ := schedReqs_rank hc _ sched cr hdg hcap hdom h0
      (by omega) (by omega) h2 hdone
    have hcs : ∀ k < alg.size, (cr.getD k #[]).size = cnt t.st.m k := fun k hk' => by
      rw [(hsz' k hk').1, hg k hk']
      exact hcnt k hk'
    have hpos : ∀ k < alg.size, 0 < (cr.getD k #[]).size := fun k hk' => by
      rw [(hsz' k hk').1]
      exact (hsz' k hk').2
    obtain ⟨P, hP, hrk⟩ := hRk ms hmask
    have hcz : (_root_.Flock.Firewall.Frame.run c {} (schedReqs
        (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched cr
          alg.size)).1.m.coins.toList.length = 2 * t.st.m + t.st.fold.kLog - 15 := by
      rw [Array.length_toList, claimPoints_size hP, hlay]
      simp only [_root_.Flock.Zk.PadLayout.new, K_SKIP, N_INNER]
      omega
    have hfa : (flatR (fun k => cr.getD k #[]) alg.size).length = 2 * t.st.m + t.st.fold.kLog - 15 := by
      rw [show alg.size = (alg.size - 2) + 2 by omega, flatR_len hcs (alg.size - 2) (by omega)]
      omega
    have hdA : d = alg.size := by
      rcases Nat.lt_or_ge d alg.size with hlt | hge
      · have := flatR_lt_of_pos (fun k => cr.getD k #[]) hlt (hpos d hlt)
        rw [← hco] at this
        omega
      · omega
    subst hdA
    refine ⟨hA, hcs, _, P, hco, hP, hrk, fun h => ?_, fun h hne => ?_⟩
    · rw [ite_eq_left h] at hp1
      rw [hpr]
      exact hp1 rfl
    · rw [ite_eq_right h] at hp2
      rw [hpr]
      exact hp2 hne
  · rw [h2] at ha
    simp only [↓reduceIte] at ha
    exact absurd ha (answer_owed_none cs rfl)

end Ranked

namespace LeanRelease

open Claims Claims.ZKMal
open _root_.Flock (V F128)
open _root_.Flock.Firewall.Frame (K_SKIP)
open Loop Ranked

set_option maxHeartbeats 800000 in
/-- **The frame refuses a rank-deficient rep at its opening's label**: on a live history of at most `refK x` openings
whose last coins the frame takes, past table `j`'s rep `r`'s rank check (`Masked.rankSlot`), the rep's rank check holds
at the opened coins. The round at the rank check is the opening's first, whose label runs the frame's rank check on the
coins the stream answered (`Masker.label`); the frame took that round's coins, where the rank check held
(`repState_rank_pcs`) on the algebraic phase's coins, which are the opened ones (`wordsAt_slot`), and, at rep `1`, after
rep `0`'s claim points, whose rounds the loop asks for the same counts (`loopNr_rep`). Both read as the session's
(`frame_points`), as at the rep's `β` round (`rank_unrefused`). -/
theorem rank_unrefused_sched (T : ZKMal.CoinLayer.TableData)
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
    {j r : ℕ} (hj : j < (Masked.stsOf x).size) (hr2 : r < 2) (hN : Masked.rankSlot x j r < ops.length) :
    ¬ Masked.RankBad (x := x) ((ZKMal.CoinLayer.session loopNr T).pad (K := refK x)
      (ops.map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2)) j r := by
  intro hbad
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨y⟩ := hx.2 j hj
  obtain ⟨-, -, hspec, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hq2 : 2 * j + r < spec.repStreams.size := by rw [FlockDeploy.repStreams_size, hspec]; omega
  have hjj : (2 * j + r) / 2 = j := by omega
  have hst : (Masked.stsOf x)[(2 * j + r) / 2]! = (Masked.stsOf x)[j] := by
    rw [hjj]
    exact getElem!_pos _ j hj
  have hstj : (Masked.stsOf x)[j]! = (Masked.stsOf x)[j] := getElem!_pos _ j hj
  have hk6 : 6 ≤ (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog := by rw [hst]; exact y.wf.1.hk6
  have hm14 : 14 ≤ (Masked.stsOf x)[j].m := by have := (hall j hj).1; omega
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j + r) / 2]!.m = .ok (y.sch, y.tPad) := by
    rw [hst]; exact y.hz
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
  obtain ⟨A, hAd⟩ : ∃ A, A = (Masked.stsOf x)[j]!.m + (Masked.stsOf x)[j]!.c.kLog - 6 := ⟨_, rfl⟩
  have hAN : A < N := by
    rw [hNd, hAd, hst, hstj]
    unfold FlockComplete.piopRoundsZk
    omega
  obtain ⟨s, hs⟩ : ∃ s, s = FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) + A := ⟨_, rfl⟩
  have hbs : Masked.rankSlot x j r = s := by
    unfold Masked.rankSlot
    rw [Masked.repBase_eq hsx, hs, hAd]
  rw [hbs] at hN
  have hlast : (ofLoop T L₀ real).last x = refK x + 1 := loop_last loopNr T (withFrames T L₀ real) x
  obtain ⟨-, m, ha⟩ := coins_taken_of_live (ofLoop T L₀ real) wk x w ρ κ ((m₀, ρ₀) :: ops) hlive
    (by rw [hlast]; simp only [List.length_cons]; omega) hr (i := s + 2) (by omega)
    (by simp only [List.length_cons]; omega)
  obtain ⟨⟨dm, da⟩, hd⟩ : ∃ d, ops[s]? = some d := ⟨ops[s]'(by omega), List.getElem?_eq_getElem _⟩
  have htk : ops.take (s + 1) = ops.take s ++ [(dm, da)] := by rw [List.take_add_one, hd]; rfl
  have hh : ((m₀, ρ₀) :: ops).take (s + 2) = (m₀, ρ₀) :: (ops.take s ++ [(dm, da)]) := by
    rw [List.take_succ_cons, htk]
  have hts : (ops.take s).length = s := by rw [List.length_take]; omega
  rw [hh] at ha
  change (progOf x ((ops.take s ++ [(dm, da)]).map fun p => ZKMal.CoinLayer.rd p.2)).answer _
    (coinsAt x ((m₀, ρ₀) :: (ops.take s ++ [(dm, da)]))) = .ok m at ha
  rw [List.map_append, List.map_cons, List.map_nil] at ha
  have hloc : ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (s - 1) = (2 * j + r, A) := by
    have e := locate_take_sum (roundsOf spec (Masked.stsOf x)) 0 (2 * j + r) A (by rw [hrsl]; exact hq2)
      (by rw [hrq _ rfl hq2]; omega)
    rw [Nat.zero_add] at e
    rw [← e, hs, repStartZk_eq spec _ hq2.le]
    congr 1
    omega
  rw [progOf_snoc x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2) (ZKMal.CoinLayer.rd da) hsx (q := s - 1)
    (by rw [List.length_map, hts]; omega), hloc] at ha
  dsimp only at ha
  rcases hph : placeholder spec (Masked.stsOf x) (2 * j + r)
      (wordsAt x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2) 0) with e | ⟨c', t, dom, xs, hi⟩
  · rw [hph] at ha
    exact absurd ha (answer_owed_none _ rfl)
  rw [hph] at ha
  dsimp only at ha
  obtain ⟨CR0, CR, hCR0, hCR, ha⟩ := repState_args ha
  obtain ⟨alg, -, hal, -⟩ := repState_ok ha
  obtain ⟨hts', ⟨tp, htp⟩, hxs⟩ := Count.placeholder_tab hph
  rw [hz, Except.ok.injEq, Prod.mk.injEq] at htp
  obtain ⟨htsch, -⟩ := htp
  obtain ⟨hm36, hk0, hn, hq⟩ := Count.sched_facts (htsch ▸ hz : _root_.Flock.Zk.schedule _ = .ok (t.sch, y.tPad))
  obtain ⟨-, hdom', hxr, -⟩ := sizesOk_at (sizesOk_of_serves hSv s0 spec _ hsx) hq2
  obtain ⟨lay, mask, hds, hc', hdm, -⟩ := placeholder_ok hph
  have hls : (wordsAt x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2) 0).size =
      min (loopNr x 0) _root_.Flock.Firewall.Frames.COIN_MAX := by
    obtain ⟨p0, rest, hpr⟩ : ∃ p0 rest, ops.take s = p0 :: rest := by
      cases h' : ops.take s with
      | nil => rw [h', List.length_nil] at hts; omega
      | cons p0 rest => exact ⟨p0, rest, rfl⟩
    rw [hpr]
    simp [wordsAt, FlockDeploy.Coin.vals]
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
  have hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by
    rw [hts']
    show (Masked.stsOf x)[(2 * j + r) / 2]!.merkle = _
    rw [hst]
    exact (hall j hj).2.2
  obtain ⟨sch', tp', -, hlay0, hms⟩ := start_eq hds
  have hmask : c'.mask = some mask := by rw [hc']; rfl
  have htm : t.st.m = (Masked.stsOf x)[j]!.m := by rw [hts', hst, hstj]; rfl
  have htk : t.st.fold.kLog = (Masked.stsOf x)[j]!.c.kLog := by rw [hts', hst, hstj]; rfl
  have hlay : c'.lay = _root_.Flock.Zk.PadLayout.new t.st.m t.st.fold.kLog sch'.k0 := by
    rw [hc']
    show lay = _
    rw [hlay0, hts']
    rfl
  have hkl : c'.lay.nLc + K_SKIP = (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog := by
    rw [hc']
    show lay.nLc + K_SKIP = _
    rw [hlay0]
    show _ - 6 + 6 = _
    omega
  have hcc : ∀ p, p < c'.order.size → c'.order[p]! < c'.h.size := by rw [hc']; exact ofDraws_order _ _ _
  have hm14t : 14 ≤ t.st.m := by rw [htm, hstj]; exact hm14
  have hk6t : 6 ≤ t.st.fold.kLog := by rw [htk, hstj]; exact y.wf.1.hk6
  have hAA : alg.size = A := by rw [(alg_cnt hm14t hk6t hmb hal).1, htm, htk, hAd]
  rw [show A + 1 = alg.size + 1 by omega] at ha
  have hCRs : CR.size = alg.size + 1 := by
    rw [hCR, Array.size_push, Array.size_map, Array.size_range, hAA]
  obtain ⟨hA', hcs, S, P, hco, hP, hrk, hp0, hp1⟩ := repState_rank_pcs hcc hal hm14t hk6t hmb hdg
    (placeholder_cap hph) (hdm ▸ hdom') hmk hxb hk0 hn hq hCRs hmask hlay ha
  rw [htm, htk] at hA' hlay
  rw [htm] at hcs
  have hm14' : 14 ≤ (Masked.stsOf x)[j]!.m := by rw [hstj]; exact hm14
  have hk6' : 6 ≤ (Masked.stsOf x)[j]!.c.kLog := by rw [hstj]; exact y.wf.1.hk6
  have hW : ∀ i < A, CR.getD i #[] =
      wordsAt x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2)
        (FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) + i) := by
    intro i hi'
    rw [hCR, crAt_getD _ _ _ hi']
    generalize hR0 : FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) = R0
    rw [show s - 1 - A + i + 1 = R0 + i by omega]
  have hok := Rank.maskRankOk_of_maskRank (S.prior ++ P) mask (c'.lay.nLc + K_SKIP) (by rw [hrk]; omega)
  rw [hkl, Rank.words_maskSlot hms, Array.map_append, hjj] at hok
  obtain ⟨P', hP', hPm⟩ := frame_points sch'.k0 hm14' hk6' (fun k => CR.getD k #[]) S.coins (hA' ▸ hco)
    (hA' ▸ hcs) (fun n e => Masked.slotAt (x := x) ((ZKMal.CoinLayer.session loopNr T).pad (K := refK x)
      (ops.map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2)) (Masked.repBase x (2 * j + r) + n) e)
    (fun i hi' e he => by
      rw [hW i (by omega), wordsAt_slot T x ops hl, Masked.repBase_eq hsx]
      · omega
      · omega
      · have := hcs i (by omega)
        rw [hW i (by omega)] at this
        omega)
  rw [hlay, hP'] at hP
  obtain rfl : P' = P := Except.ok.inj hP
  have hprior : S.prior.map Array.toList =
      if r = 1 then Masked.ptsOpen (x := x) ((ZKMal.CoinLayer.session loopNr T).pad (K := refK x)
        (ops.map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2)) (Masked.stsOf x)[j]! (2 * j) else #[] := by
    rcases (show r = 0 ∨ r = 1 by omega) with rfl | rfl
    · rw [hp0 (by omega)]
      simp
    · rw [ite_eq_left rfl]
      have hq0 : 2 * j < spec.repStreams.size := by omega
      have hr0 : (roundsOf spec (Masked.stsOf x)).getD (2 * j + 1 - 1) 0 = N + 4 :=
        hrq _ (by omega) (by omega)
      have hst0 := FlockComplete.ZkTables.repStart_succ spec (Masked.stsOf x) (2 * j)
      rw [show (2 * j) / 2 = (2 * j + 1) / 2 by omega, hrR _ rfl] at hst0
      have hs10 : 1 ≤ FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j) := by
        unfold FlockComplete.repStartZk; omega
      have hW0 : ∀ i < N + 4, CR0.getD i #[] =
          wordsAt x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2)
            (FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j) + i) := by
        intro i hi'
        rw [hCR0, crMap_getD _ _ (by rw [hr0]; exact hi'), hr0]
        generalize hR0 : FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j) = R0
        rw [show s - 1 - A - (N + 4) + i + 1 = R0 + i by omega]
      have hCR0s : alg.size ≤ CR0.size := by
        rw [hCR0, Array.size_map, Array.size_range, hr0]
        omega
      have hcs0 : ∀ i < alg.size, (CR0.getD i #[]).size = cnt (Masked.stsOf x)[j]!.m i := fun i hi' => by
        have h1 := hcs i hi'
        rw [hW i (by omega), wordsAt_size x ops (by omega) (by omega),
          loopNr_rep hsx (q := 2 * j + 1) (by omega) (by rw [hrq _ (by omega) (by omega)]; omega)] at h1
        rw [hW0 i (by omega), wordsAt_size x ops (by omega) (by omega),
          loopNr_rep hsx (q := 2 * j) hq0 (by rw [hrq _ (by omega) hq0]; omega),
          show (2 * j) / 2 = (2 * j + 1) / 2 by omega]
        exact h1
      have hP0 := prior_toList CR0 hCR0s
      have hne : ((CR0.extract 0 alg.size).foldl (· ++ ·) #[]).isEmpty = false := by
        refine Bool.eq_false_iff.2 fun h => ?_
        rw [Array.isEmpty_iff_size_eq_zero] at h
        have h1 := congrArg List.length hP0
        rw [Array.length_toList, h, show alg.size = (alg.size - 2) + 2 by omega,
          flatR_len (hA' ▸ hcs0) (alg.size - 2) (by omega)] at h1
        omega
      have hpc := hp1 (by omega) hne
      rw [words_wordBytes, hlay] at hpc
      obtain ⟨P0, hP0', hP0m⟩ := frame_points sch'.k0 hm14' hk6' (fun k => CR0.getD k #[]) _ (hA' ▸ hP0)
        (hA' ▸ hcs0) (fun n e => Masked.slotAt (x := x) ((ZKMal.CoinLayer.session loopNr T).pad (K := refK x)
          (ops.map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2)) (Masked.repBase x (2 * j) + n) e)
        (fun i hi' e he => by
          rw [hW0 i (by omega), wordsAt_slot T x ops hl, Masked.repBase_eq hsx]
          · omega
          · omega
          · have := hcs0 i (by omega)
            rw [hW0 i (by omega)] at this
            omega)
      rw [← hA'] at hP0'
      rw [hP0'] at hpc
      obtain rfl : P0 = S.prior := Except.ok.inj hpc
      exact hP0m
  rw [hprior, hPm] at hok
  unfold Masked.RankBad Masked.ptsOpen at hbad
  unfold Masked.ptsOpen at hok
  rw [hok] at hbad
  exact Bool.noConfusion hbad

end LeanRelease

end Claims.ZKServer
