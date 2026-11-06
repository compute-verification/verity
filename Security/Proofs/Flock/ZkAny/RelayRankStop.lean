module

public import Proofs.Flock.Claims.ZK.FrameRankFault
public import Proofs.Flock.ZkAny.RelayRank
public import Proofs.Flock.Claims.ZK.FrameOuter

/-!
# Where the model's rank stop falls in a rep

The model's rank stop (`LeanRelease.rankFails`) fails only at a state in the algebraic phase holding all its coins, as
many as the claim points read (`2m + k_log − 15` at the rep's layout). The loop's state at a rep's round `j + 1`
holds rounds `0 … j − 1`'s coins at the algebraic phase's counts and owes round `j`'s, so the coins after it leave such
a state only where `j + 1` is the algebraic phase's round count:

* `Ranked.answer_done`: coins the frame takes keep whether it is past the opening's label.
* `Ranked.rankFails_coins`: where the rank stop fails, the state is in the algebraic phase and holds the claim points'
  coins.
* `Ranked.answer_ite`, `Ranked.repState_pairs`, `Ranked.run_head_zero`: the loop's state at a rep's round `j + 1` is the
  frame's along the rep's first round, then each round's coins and the next; after the first round alone it owes the
  zerocheck head's `6` coins and holds none, or owes none.
* **`Ranked.repState_rankFails`**: where the coins after the loop's state at a rep's round `j + 1`, below the inner
  proof's end, leave a state at which the rank stop fails, at a configuration of the rep's layout, `j + 1` is the
  algebraic phase's round count.
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec Transcript TMW)
open _root_.Flock.Firewall.Frame (Cfg Masker MaskSlot K_SKIP N_INNER LABEL_ZC maskRank claimPoints INNER_ROUNDS)
open FlockSoundness.Refine (Frame encs)

namespace Ranked

open Loop

/-- **Coins the frame takes keep whether it is past the opening's label.** -/
theorem answer_done {c : Cfg} {s m : Masker} {cs : Array F128} (h : s.answer c cs = .ok m) : m.done = s.done := by
  cases hi : s.inner with
  | some k => exact (answer_proj (Or.inl (by simp [hi])) h).1
  | none =>
    cases hl : s.l0 with
    | false =>
      have e := answer_alg c s cs hi hl h
      subst e
      rfl
    | true => exact (answer_proj (Or.inr hl) h).1

/-- **Where the rank stop fails, the state is in the algebraic phase and holds the claim points' coins**: at the rep's
layout, `2m + k_log − 15` of them. -/
theorem rankFails_coins {c : Cfg} {s : Masker} (hr : LeanRelease.rankFails c s = true) {m kLog k0 : ℕ}
    (hlay : c.lay = _root_.Flock.Zk.PadLayout.new m kLog k0) (hm : 14 ≤ m) (hk : 6 ≤ kLog) :
    s.done = false ∧ s.on = true ∧ s.coins.toList.length = 2 * m + kLog - 15 := by
  have hon : s.on = true := by
    cases h : s.on
    · simp [LeanRelease.rankFails, h] at hr
    · rfl
  have hd : s.done = false := by
    cases h : s.done
    · rfl
    · simp [LeanRelease.rankFails, h] at hr
  refine ⟨hd, hon, ?_⟩
  cases hmk : c.mask with
  | none => simp [LeanRelease.rankFails, hmk] at hr
  | some ms =>
    cases hcp : claimPoints c.lay s.coins with
    | error e => simp [LeanRelease.rankFails, hmk, hcp] at hr
    | ok P =>
      rw [Array.length_toList, claimPoints_size hcp, hlay]
      simp only [_root_.Flock.Zk.PadLayout.new, K_SKIP, N_INNER]
      omega

/-- Coins answered at the loop's empty state fail; elsewhere they are the state's. -/
theorem answer_ite {c : Cfg} {b : Bool} {s m : Masker} {cs : Array F128}
    (h : (if b = true then ({} : Masker) else s).answer c cs = .ok m) : b = false ∧ s.answer c cs = .ok m := by
  cases b
  · exact ⟨rfl, h⟩
  · exact absurd h (answer_owed_none _ rfl)

/-- **The loop's state at a rep's round `j + 1`**, below the inner proof's end, is the frame's along the rep's first
round, then each round's coins and the next (`qOf`), after an earlier rep's coins. -/
theorem repState_pairs (c : Cfg) {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray}
    {xs : Array (Array F128)} {hi : Array ByteArray} {cr0 cr : Array (Array F128)} {ns : ℕ → ℕ}
    {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr.pop = .ok sched) (hS : 0 < sched.size) {j : ℕ}
    (hj : j < sched.size + INNER_ROUNDS) :
    ∃ own : Array (ByteArray × ℕ), ∃ P : Array F128, repState c t rep dom xs hi cr0 cr ns (j + 1) =
      let r := _root_.Flock.Firewall.Frame.run c {}
        ((if P.isEmpty then [] else [.prior P.size (_root_.Flock.Firewall.Frame.wordBytes P)]) ++
          (qOf sched own 0 :: pairs (fun k => cr.getD k #[]) (qOf sched own) j))
      if r.2 then {} else r.1.m := by
  by_cases hjs : j < sched.size
  · refine ⟨#[], if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[], ?_⟩
    rw [repState_at c t rep dom xs hi cr0 cr ns j hal hs hjs, schedReqs_pairs _ sched #[] cr j hjs]
  · obtain ⟨e, rfl⟩ : ∃ e, j = sched.size + e := ⟨j - sched.size, by omega⟩
    refine ⟨ownRounds c (cr.extract sched.size cr.size) (fun j => ns (sched.size + j)),
      if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[], ?_⟩
    rw [repState_own_at c t rep dom xs hi cr0 cr ns e hal hs (by omega), ownReqs_pairs _ sched _ cr hS e]

/-- **After a rep's first round alone** the frame owes the zerocheck head's `6` coins, holding none, in the algebraic
phase, or owes none. -/
theorem run_head_zero {c : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) (p : Array (Array F128))
    {digest cap dom : ByteArray} (hdg : digest.size < 2 ^ 64) (hcap : cap.size < 2 ^ 64) (hdom : dom.size < 2 ^ 64)
    {R : ByteArray} (hR : R = ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes digest,
      .bytes cap, .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray)
    (h2 : (_root_.Flock.Firewall.Frame.run c { m := { prior := p } } [.round 6 R (.sched R)]).2 = false) :
    (_root_.Flock.Firewall.Frame.run c { m := { prior := p } } [.round 6 R (.sched R)]).1.m.owed = none ∨
      ((_root_.Flock.Firewall.Frame.run c { m := { prior := p } } [.round 6 R (.sched R)]).1.m.owed = some 6 ∧
        (_root_.Flock.Firewall.Frame.run c { m := { prior := p } } [.round 6 R (.sched R)]).1.m.coins = #[] ∧
        (_root_.Flock.Firewall.Frame.run c { m := { prior := p } } [.round 6 R (.sched R)]).1.m.on = true ∧
        (_root_.Flock.Firewall.Frame.run c { m := { prior := p } } [.round 6 R (.sched R)]).1.m.l0 = false ∧
        (_root_.Flock.Firewall.Frame.run c { m := { prior := p } } [.round 6 R (.sched R)]).1.m.inner = none) := by
  subst hR
  rcases Loop.run_sched_cons (st := { m := { prior := p } }) rfl h2 with ⟨out, sx, why, hck, he⟩ | he
  · rw [he]
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
    right
    rcases hsx with rfl | rfl <;> exact ⟨rfl, rfl, rfl, rfl, rfl⟩
  · rw [he]
    exact Or.inl rfl

/-- **The rank stop fails after a rep's round only at the algebraic phase's end**: where the coins after the loop's
state at a rep's round `j + 1`, below the inner proof's end, leave a state at which the rank stop fails, at a
configuration of the rep's layout, `j + 1` is the algebraic phase's round count. -/
theorem repState_rankFails {c c'' : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size)
    {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} {cr0 cr : Array (Array F128)} {ns : ℕ → ℕ} {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr.pop = .ok sched) (hm : 14 ≤ t.st.m) (hk : 6 ≤ t.st.fold.kLog)
    (hm' : t.st.m < 2 ^ 64) (hdg : t.st.digest.size < 2 ^ 64)
    (hcap : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64) (hdom : dom.size < 2 ^ 64) {k0 : ℕ}
    (hlay : c''.lay = _root_.Flock.Zk.PadLayout.new t.st.m t.st.fold.kLog k0) {j : ℕ}
    (hj : j < sched.size + INNER_ROUNDS) {cs : Array F128} {m : Masker}
    (hA : (repState c t rep dom xs hi cr0 cr ns (j + 1)).answer c'' cs = .ok m)
    (hr : LeanRelease.rankFails c'' m = true) : j + 1 = alg.size := by
  obtain ⟨hd, -, hlen⟩ := rankFails_coins hr hlay hm hk
  obtain ⟨ha0, -⟩ := Sched.rounds_two hm hm' hal
  obtain ⟨rest, hrest⟩ := schedSoFar_pre hal hs
  have hg : ∀ k, k < alg.size → sched.getD k (.empty, 0) = (alg[k]?).getD (.empty, 0) := fun k hk => by
    rw [hrest, Array.getD_eq_getD_getElem?, Array.getElem?_append_left hk]
  obtain ⟨hAl, hcnt⟩ := alg_cnt hm hk hm' hal
  have hAS : alg.size ≤ sched.size := by rw [hrest, Array.size_append]; omega
  have hS : 0 < sched.size := by omega
  have h0 : sched.getD 0 (.empty, 0) = (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8,
      .bytes t.st.digest, .bytes (_root_.Flock.Firewall.Finish.flat t.cm0.cap),
      .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray, 6) := by
    rw [hg 0 (by omega), ha0, Option.getD_some, head_encs]
  obtain ⟨own, P, hst⟩ := repState_pairs c hal hs hS hj
  have hnn : ∀ k < alg.size, nnOf sched own k = cnt t.st.m k := fun k hk' => by
    simp only [nnOf, show k < sched.size by omega, ↓reduceIte]
    rw [hg k hk']
    exact hcnt k hk'
  rw [hst] at hA
  dsimp only at hA
  obtain ⟨h2, hA⟩ := answer_ite hA
  obtain ⟨p, -, -, he⟩ := run_prior_eq c P _ h2
  rw [he] at h2 hA
  have hq0 : qOf sched own 0 = .round 6 (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8,
      .bytes t.st.digest, .bytes (_root_.Flock.Firewall.Finish.flat t.cm0.cap), .label LABEL_ZC.toUTF8] ++
      (Frame.sq 2 6).enc).toArray)
      (.sched (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes t.st.digest,
        .bytes (_root_.Flock.Firewall.Finish.flat t.cm0.cap), .label LABEL_ZC.toUTF8] ++
        (Frame.sq 2 6).enc).toArray)) := by
    simp only [qOf, hS, ↓reduceIte, h0]
  rw [hq0] at h2 hA
  rcases Nat.eq_zero_or_pos j with rfl | hj0
  · exfalso
    simp only [pairs, List.range_zero, List.flatMap_nil] at h2 hA
    rcases run_head_zero hc p hdg hcap hdom rfl h2 with ho | ⟨ho, hco, hon, hl, hi'⟩
    · exact answer_owed_none _ ho hA
    · have hsz := answer_size ho hA
      have e := answer_alg c'' _ cs hi' hl hA
      subst e
      simp only [hon, hco, ↓reduceIte, Array.empty_append, Array.length_toList] at hlen
      omega
  · obtain ⟨hg', -, hsz, ho, d, -, hco, hdk⟩ := run_head hc p hdg hcap hdom (qOf_round sched own)
      (by simp only [nnOf, hS, ↓reduceIte, h0]) (J := j) hj0 h2
    have hn := ho.resolve_left fun h => answer_owed_none _ h hA
    have hsd := (answer_done hA).symm.trans hd
    obtain ⟨hon, hl, hi'⟩ := hg'.1 hsd
    rw [hdk hsd] at hco
    have hcs := answer_size hn.1 hA
    have hn1 := hn.2
    have e := answer_alg c'' _ cs hi' hl hA
    subst e
    simp only [hon, ↓reduceIte, Array.toList_append, List.length_append, hco, Array.length_toList] at hlen
    rcases Nat.lt_or_ge j alg.size with hja | hja
    · have hF : ∀ i < j, (cr.getD i #[]).size = cnt t.st.m i := fun i hi => (hsz i hi).1.trans (hnn i (by omega))
      rw [hcs, hnn j hja] at hlen
      rcases Nat.lt_or_ge j 2 with hj2 | hj2
      · obtain rfl : j = 1 := by omega
        have h1 : cnt t.st.m 1 = t.st.m - 13 := by simp [cnt]
        rw [flatR_len_one hF (le_refl 1), h1] at hlen
        omega
      · obtain ⟨n, rfl⟩ : ∃ n, j = n + 2 := ⟨j - 2, by omega⟩
        rw [flatR_len hF n (le_refl _), cnt_two hj2] at hlen
        omega
    · have hF : ∀ i < alg.size, (cr.getD i #[]).size = cnt t.st.m i := fun i hi =>
        (hsz i (by omega)).1.trans (hnn i hi)
      have h1 := flatR_len hF (alg.size - 2) (by omega)
      rw [show alg.size - 2 + 2 = alg.size by omega] at h1
      have h3 := flatR_mono (fun k => cr.getD k #[]) hja
      omega

end Ranked

end Claims.ZKServer
