import Proofs.Flock.Claims.ZK.FrameRanked
import Proofs.Flock.Claims.ZK.FrameBeta

/-!
# The frames refuse a rank-deficient mask slot (`FrameUnrefused`'s rank fault)

At a `Live` statement the loop serves, on a live history of `refK x` openings whose last coins the frame takes, no
table's rep's rank check refuses at the opened coins (`Masked.RankBad`, `Masked.Thrown`'s first disjunct). The opening at
the rep's inner round 2 (`betaSlot`) finds the frame in the inner proof, past the opening's label, where the rank check
held (`Ranked.ownReqs_rank`); what it read is the session's:

* `Ranked.cnt`, `Ranked.alg_cnt`: the algebraic phase's rounds answer `6`, `m − 13`, then one coin each.
* `Ranked.flatR_len`, `Ranked.flatR_getElem?`, **`Ranked.frame_points`**: on the rounds' coins laid end to end, at those
  counts, the frame's claim points are `MaskRank.claimPoints` at the rounds (`Rank.claimPoints_frame`).
* `Ranked.words_wordBytes`, `Ranked.foldl_append_toList`: an earlier rep's coins, sent as bytes, read back as its
  rounds' coins laid end to end.
* **`Ranked.repState_rank`**: at the loop's state at a rep's inner round 2, the frame held the rounds' coins, at the
  algebraic phase's counts, and the rank check held on them after the earlier rep's points.
* `Loop.wordsAt_slot`, `Loop.wordsAt_size`, `Loop.loopNr_rep`: the coins the loop hands a rep's state are the opened
  coins, as many as the rep's plan asks.
* **`LeanRelease.rank_unrefused`**: the fault.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec Transcript TMW)
open _root_.Flock.Firewall.Frame (Cfg Masker MaskSlot K_SKIP N_INNER LABEL_ZC maskRank claimPoints INNER_ROUNDS)
open FlockSoundness.Refine (Frame encs)

namespace Ranked

open Loop

/-- **The algebraic phase's count at its round `k`**: the zerocheck head's `6`, the outer round's `m − 13`, then one. -/
def cnt (m k : ℕ) : ℕ := if k = 0 then 6 else if k = 1 then m - 13 else 1

theorem cnt_two {m k : ℕ} (h : 2 ≤ k) : cnt m k = 1 := by
  unfold cnt
  rw [ite_eq_right (by omega), ite_eq_right (by omega)]

theorem counts_eq {m kLog : ℕ} (hm : 14 ≤ m) (hk : 6 ≤ kLog) :
    Count.zcCounts m ++ Count.lcCounts kLog = 6 :: (m - 13) :: List.replicate (m + kLog - 8) 1 := by
  have e : ∀ n, (List.replicate n [1]).flatten = List.replicate n 1 := fun n => by
    induction n with
    | zero => rfl
    | succ n ih => rw [List.replicate_succ, List.flatten_cons, ih]; rfl
  rw [show m + kLog - 8 = 1 + (m - 6) + 1 + 1 + (kLog - 6) + 1 by omega]
  simp [Count.zcCounts, Count.lcCounts, Count.sq, e, show m - 13 ≠ 0 by omega, List.replicate_add]

/-- **The algebraic phase's rounds answer `6`, `m − 13`, then one coin each**, `m + k_log − 6` of them. -/
theorem alg_cnt {st : _root_.Flock.Setup} {k0 : ℕ} {cap dom : ByteArray} {alg : Array (ByteArray × ℕ)}
    (hm : 14 ≤ st.m) (hk : 6 ≤ st.fold.kLog) (hm' : st.m < 2 ^ 64)
    (h : _root_.Flock.Firewall.Skeleton.rounds st k0 cap dom = .ok alg) :
    alg.size = st.m + st.fold.kLog - 6 ∧ ∀ k < alg.size, ((alg[k]?).getD (.empty, 0)).2 = cnt st.m k := by
  have e := Count.rounds_counts hm' h
  rw [counts_eq hm hk] at e
  have hs : alg.size = st.m + st.fold.kLog - 6 := by
    have := congrArg List.length e
    simp only [List.length_map, Array.length_toList, List.length_cons, List.length_replicate] at this
    omega
  refine ⟨hs, fun k hk' => ?_⟩
  have e2 := congrArg (fun l => l[k]?) e
  simp only [List.getElem?_map, Array.getElem?_toList] at e2
  rw [Array.getElem?_eq_getElem hk'] at e2 ⊢
  simp only [Option.map_some, Option.getD_some] at e2 ⊢
  rcases k with _ | _ | k
  · simpa [cnt] using e2
  · simpa [cnt] using e2
  · have hk2 : k < st.m + st.fold.kLog - 8 := by omega
    simp [cnt, hk2] at e2 ⊢
    exact e2

theorem flatR_length (cs : ℕ → Array F128) : ∀ d, (flatR cs d).length = ((List.range d).map fun i => (cs i).size).sum
  | 0 => rfl
  | d + 1 => by
    rw [flatR_succ, List.length_append, flatR_length cs d, List.range_succ, List.map_append, List.sum_append,
      Array.length_toList]
    simp

theorem flatR_mono (cs : ℕ → Array F128) {k : ℕ} : ∀ {d : ℕ}, k ≤ d → (flatR cs k).length ≤ (flatR cs d).length
  | 0, h => by rw [Nat.le_zero.1 h]
  | d + 1, h => by
    rcases Nat.lt_or_ge k (d + 1) with hk | hk
    · rw [flatR_succ, List.length_append]
      exact Nat.le_add_right_of_le (flatR_mono cs (by omega))
    · rw [show k = d + 1 by omega]

theorem flatR_lt (cs : ℕ → Array F128) {k d e : ℕ} (hk : k < d) (he : e < (cs k).size) :
    (flatR cs k).length + e < (flatR cs d).length := by
  have h := flatR_mono cs (show k + 1 ≤ d by omega)
  have e' : (cs k).toList.length = (cs k).size := Array.length_toList
  rw [flatR_succ, List.length_append, e'] at h
  omega

theorem flatR_getElem? (cs : ℕ → Array F128) : ∀ {d k e : ℕ}, k < d → (he : e < (cs k).size) →
    (flatR cs d)[(flatR cs k).length + e]? = some (cs k)[e]
  | 0, _, _, h, _ => absurd h (Nat.not_lt_zero _)
  | d + 1, k, e, h, he => by
    rw [flatR_succ]
    rcases Nat.lt_or_ge k d with hk | hk
    · rw [List.getElem?_append_left (flatR_lt cs hk he)]
      exact flatR_getElem? cs hk he
    · obtain rfl : k = d := by omega
      rw [List.getElem?_append_right (by omega), Nat.add_sub_cancel_left, Array.getElem?_toList,
        Array.getElem?_eq_getElem he]

/-- Past round `1`, rounds `0 … n + 1` at the algebraic phase's counts hold `6 + (m − 13) + n` coins. -/
theorem flatR_len {cs : ℕ → Array F128} {m D : ℕ} (hs : ∀ i < D, (cs i).size = cnt m i) :
    ∀ n, n + 2 ≤ D → (flatR cs (n + 2)).length = 6 + (m - 13) + n
  | 0, h => by
    show (flatR cs (0 + 1 + 1)).length = _
    have e0 : (cs 0).toList.length = (cs 0).size := Array.length_toList
    have e1 : (cs 1).toList.length = (cs 1).size := Array.length_toList
    rw [flatR_succ, flatR_succ, List.length_append, List.length_append, e0, e1, hs 0 (by omega), hs 1 (by omega)]
    simp [flatR, cnt]
  | n + 1, h => by
    have e : (cs (n + 2)).toList.length = (cs (n + 2)).size := Array.length_toList
    rw [show n + 1 + 2 = (n + 2) + 1 by omega, flatR_succ, List.length_append, flatR_len hs n (by omega), e,
      hs (n + 2) (by omega), cnt_two (by omega)]
    omega

theorem flatR_len_one {cs : ℕ → Array F128} {m D : ℕ} (hs : ∀ i < D, (cs i).size = cnt m i) (hD : 1 ≤ D) :
    (flatR cs 1).length = 6 := by
  show (flatR cs (0 + 1)).length = _
  have e0 : (cs 0).toList.length = (cs 0).size := Array.length_toList
  rw [flatR_succ, List.length_append, e0, hs 0 (by omega)]
  simp [flatR, cnt]

/-- **The frame's claim points at the rounds' coins laid end to end**: on `m + k_log − 6` rounds at the algebraic
phase's counts, whose coins are `coin`'s, `Frame.claimPoints` at the rep's layout returns `MaskRank.claimPoints` at
`coin`, as lists (`Rank.claimPoints_frame`). -/
theorem frame_points {m kLog : ℕ} (k0 : ℕ) (hm : 14 ≤ m) (hk : 6 ≤ kLog) (cs : ℕ → Array F128) (a : Array F128)
    (ha : a.toList = flatR cs (m + kLog - 6)) (hs : ∀ i < m + kLog - 6, (cs i).size = cnt m i)
    (coin : ℕ → ℕ → F128) (hv : ∀ i < m + kLog - 6, ∀ e < cnt m i, (cs i)[e]! = coin i e) :
    ∃ P, claimPoints (_root_.Flock.Zk.PadLayout.new m kLog k0) a = .ok P ∧
      P.map Array.toList = FlockComplete.MaskRank.claimPoints m kLog coin := by
  have hget : ∀ i < m + kLog - 6, ∀ e < cnt m i, ∀ n, n = (flatR cs i).length + e → a[n]! = coin i e := by
    intro i hi e he n hn
    have h := flatR_getElem? cs hi (e := e) (by rw [hs i hi]; exact he)
    rw [← ha, Array.getElem?_toList, ← hn] at h
    rw [getElem!_def, h]
    dsimp only
    rw [← getElem!_pos (cs i) e (by rw [hs i hi]; exact he)]
    exact hv i hi e he
  have hlen : ∀ i, 2 ≤ i → i ≤ m + kLog - 6 → (flatR cs i).length = 6 + (m - 13) + (i - 2) := fun i h1 h2 => by
    have := flatR_len hs (i - 2) (by omega)
    rwa [show i - 2 + 2 = i by omega] at this
  have hsz : a.size = 2 * m + kLog - 15 := by
    have := congrArg List.length ha
    rw [Array.length_toList, hlen _ (by omega) le_rfl] at this
    omega
  refine Rank.claimPoints_frame k0 (by omega) hk a hsz coin (fun e he => ?_) (fun j hj => ?_) (fun j hj => ?_)
  · exact hget 1 (by omega) e (by simp [cnt]; exact he) _ (by rw [flatR_len_one hs (by omega)])
  · exact hget (3 + j) (by omega) 0 (by rw [cnt_two (by omega)]; omega) _
      (by rw [hlen _ (by omega) (by omega)]; omega)
  · exact hget (5 + (m - 6) + j) (by omega) 0 (by rw [cnt_two (by omega)]; omega) _
      (by rw [hlen _ (by omega) (by omega)]; omega)

theorem toBytes_dsize (a : F128) : a.toBytes.data.size = 16 := by simp [F128.toBytes]

/-- **An array's words, sent as bytes, read back.** -/
theorem words_wordBytes (P : Array F128) :
    _root_.Flock.Firewall.Frame.words (_root_.Flock.Firewall.Frame.wordBytes P) 0 P.size = P := by
  refine Array.ext (by rw [words_size]) fun i h1 h2 => ?_
  simp only [_root_.Flock.Firewall.Frame.words, Array.getElem_map, Array.getElem_range]
  have hd := wordBytes_data P
  have hsplit : P.toList = P.toList.take i ++ P[i] :: P.toList.drop (i + 1) := by
    conv_lhs => rw [← List.take_append_drop i P.toList]
    rw [List.drop_eq_getElem_cons (by simpa using h2)]
    simp
  have hb : _root_.Flock.Firewall.Frame.wordBytes P =
      ⟨((P.toList.take i).flatMap fun a => a.toBytes.data.toList).toArray⟩ ++ _root_.Flock.le64 P[i].lo ++
        (_root_.Flock.le64 P[i].hi ++ ⟨((P.toList.drop (i + 1)).flatMap fun a => a.toBytes.data.toList).toArray⟩) := by
    apply ByteArray.ext
    apply Array.toList_inj.1
    conv_lhs => rw [hd, FlockSoundness.Refine.colBytes, hsplit, List.flatMap_append, List.flatMap_cons]
    simp [FlockSoundness.Discharge.ZkView.toBytes_split]
  have hp : (⟨((P.toList.take i).flatMap fun a => a.toBytes.data.toList).toArray⟩ : ByteArray).size = 16 * i := by
    show ((P.toList.take i).flatMap fun a => a.toBytes.data.toList).toArray.size = 16 * i
    simp only [List.size_toArray, List.length_flatMap, Array.length_toList, toBytes_dsize, List.map_const',
      List.sum_replicate, List.length_take, smul_eq_mul]
    omega
  have hp8 : (⟨((P.toList.take i).flatMap fun a => a.toBytes.data.toList).toArray⟩ ++ _root_.Flock.le64 P[i].lo :
      ByteArray).size = 16 * i + 8 := by
    rw [ByteArray.size_append, hp, FlockSoundness.Discharge.ZkView.le64_size]
  rw [hb, Nat.zero_add, ← hp8, ← hp, FlockSoundness.Discharge.ZkView.u64OfLE_mid, ← ByteArray.append_assoc,
    FlockSoundness.Discharge.ZkView.u64OfLE_mid]

theorem foldl_append_toList : ∀ (l : List (Array F128)) (acc : Array F128),
    (l.foldl (· ++ ·) acc).toList = acc.toList ++ l.flatMap Array.toList
  | [], acc => by simp
  | a :: l, acc => by
    rw [List.foldl_cons, foldl_append_toList l (acc ++ a), Array.toList_append, List.flatMap_cons, List.append_assoc]

theorem take_flatMap (cr : Array (Array F128)) :
    ∀ n, n ≤ cr.size → (cr.toList.take n).flatMap Array.toList = flatR (fun i => cr.getD i #[]) n
  | 0, _ => by simp [flatR]
  | n + 1, h => by
    rw [List.take_add_one, List.flatMap_append, take_flatMap cr n (by omega), flatR_succ, Array.getElem?_toList,
      Array.getElem?_eq_getElem (show n < cr.size by omega)]
    simp [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (show n < cr.size by omega)]

/-- **An earlier rep's coins laid end to end**: its first `n` rounds'. -/
theorem prior_toList (cr : Array (Array F128)) {n : ℕ} (hn : n ≤ cr.size) :
    ((cr.extract 0 n).foldl (· ++ ·) #[]).toList = flatR (fun i => cr.getD i #[]) n := by
  rw [← Array.foldl_toList, foldl_append_toList, Array.toList_extract, List.extract_eq_take_drop, List.drop_zero,
    Nat.sub_zero]
  simp only [List.nil_append]
  exact take_flatMap cr n hn

theorem claimPoints_size {l : _root_.Flock.Zk.PadLayout} {c : Array F128} {P : Array (Array F128)}
    (h : claimPoints l c = .ok P) : c.size = K_SKIP + (l.nMlv - N_INNER) + 1 + l.nMlv + 2 + l.nLc + 1 := by
  unfold claimPoints at h
  dsimp only at h
  split at h
  · cases h
  · rename_i hn
    simp only [Bool.or_eq_true, decide_eq_true_eq, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hn
    exact hn.2

theorem flatR_lt_of_pos (cs : ℕ → Array F128) {k d : ℕ} (hk : k < d) (hp : 0 < (cs k).size) :
    (flatR cs k).length < (flatR cs d).length := by
  have := flatR_lt cs hk hp
  omega

/-- **At a rep's inner round 2 the rank check held on the algebraic phase's coins**: where the loop's state there takes
coins, the frame held the rounds `0 … m + k_log − 7`'s coins (`cr`), answered at the algebraic phase's counts, and the
rank check held at their claim points, after the earlier rep's (`cr0`'s rounds, sent as bytes) at rep `1`. -/
theorem repState_rank {c c'' : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size)
    {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} {cr0 cr : Array (Array F128)} {ns : ℕ → ℕ} {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr.pop = .ok sched) (hm : 14 ≤ t.st.m) (hk : 6 ≤ t.st.fold.kLog)
    (hm' : t.st.m < 2 ^ 64) (hdg : t.st.digest.size < 2 ^ 64)
    (hcap : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64) (hdom : dom.size < 2 ^ 64)
    {ms : MaskSlot} (hmask : c.mask = some ms) {k0 : ℕ}
    (hlay : c.lay = _root_.Flock.Zk.PadLayout.new t.st.m t.st.fold.kLog k0) {cs : Array F128} {m : Masker}
    (ha : (repState c t rep dom xs hi cr0 cr ns (sched.size + 2 + 1)).answer c'' cs = .ok m) :
    alg.size = t.st.m + t.st.fold.kLog - 6 ∧ (∀ k < alg.size, (cr.getD k #[]).size = cnt t.st.m k) ∧
      ∃ S : Masker, ∃ P, S.coins.toList = flatR (fun k => cr.getD k #[]) alg.size ∧ claimPoints c.lay S.coins = .ok P ∧
        maskRank (S.prior ++ P) ms (c.lay.nLc + K_SKIP) = 128 * (S.prior ++ P).size ∧
        (rep = 0 → S.prior = #[]) ∧
        (rep ≠ 0 → ((cr0.extract 0 alg.size).foldl (· ++ ·) #[]).isEmpty = false →
          claimPoints c.lay (_root_.Flock.Firewall.Frame.words
            (_root_.Flock.Firewall.Frame.wordBytes ((cr0.extract 0 alg.size).foldl (· ++ ·) #[])) 0
            ((cr0.extract 0 alg.size).foldl (· ++ ·) #[]).size) = .ok S.prior) := by
  obtain ⟨ha0, -⟩ := Sched.rounds_two hm hm' hal
  obtain ⟨rest, hrest⟩ := schedSoFar_pre hal hs
  have hg : ∀ k, k < alg.size → sched.getD k (.empty, 0) = (alg[k]?).getD (.empty, 0) := fun k hk => by
    rw [hrest, Array.getD_eq_getD_getElem?, Array.getElem?_append_left hk]
  obtain ⟨hA, hcnt⟩ := alg_cnt hm hk hm' hal
  have hAS : alg.size ≤ sched.size := by rw [hrest, Array.size_append]; omega
  have h0 : sched.getD 0 (.empty, 0) = (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8,
      .bytes t.st.digest, .bytes (_root_.Flock.Firewall.Finish.flat t.cm0.cap),
      .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray, 6) := by
    rw [hg 0 (by omega), ha0, Option.getD_some, head_encs]
  rw [repState_own_at c t rep dom xs hi cr0 cr ns 2 hal hs (by decide)] at ha
  dsimp only at ha
  generalize hR : _root_.Flock.Firewall.Frame.run c {}
    (ownReqs (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched
      (ownRounds c (cr.extract sched.size cr.size) fun j => ns (sched.size + j)) cr 2) = R at ha
  obtain ⟨S0, b⟩ := R
  cases b with
  | true =>
    simp only [↓reduceIte] at ha
    exact absurd ha (answer_owed_none cs rfl)
  | false =>
    simp only [Bool.false_eq_true, ↓reduceIte] at ha
    have hI := run_ownReqs_inner c _ sched _ cr ⟨_, zeros 16, zeros 64, _, zeros_size 16, zeros_size 64,
      ownRounds_getD c _ _ (show 0 < INNER_ROUNDS by decide)⟩ 2 (by rw [hR])
      (by rw [hR]; exact fun e => answer_owed_none cs e ha)
    rw [hR] at hI
    obtain ⟨p, hp1, hp2, hRk, hpr, d, hd, hco, hsz⟩ := ownReqs_rank hc
      (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched
      (ownRounds c (cr.extract sched.size cr.size) fun j => ns (sched.size + j)) cr hdg hcap hdom h0 2
      (by rw [hR]) (by rw [hR]; dsimp only; rw [hI]; rfl)
    rw [hR] at hRk hpr hco
    dsimp only at hRk hpr hco
    have hcs : ∀ k < alg.size, (cr.getD k #[]).size = cnt t.st.m k := fun k hk' => by
      rw [(hsz k (by omega)).1]
      simp only [nnOf, show k < sched.size by omega, ↓reduceIte]
      rw [hg k hk']
      exact hcnt k hk'
    have hpos : ∀ k < sched.size + 2, 0 < (cr.getD k #[]).size := fun k hk' => by
      rw [(hsz k hk').1]
      exact (hsz k hk').2
    obtain ⟨P, hP, hrk⟩ := hRk ms hmask
    have hcz : S0.m.coins.toList.length = 2 * t.st.m + t.st.fold.kLog - 15 := by
      rw [Array.length_toList, claimPoints_size hP, hlay]
      simp only [_root_.Flock.Zk.PadLayout.new, K_SKIP, N_INNER]
      omega
    have hfa : (flatR (fun k => cr.getD k #[]) alg.size).length = 2 * t.st.m + t.st.fold.kLog - 15 := by
      rw [show alg.size = (alg.size - 2) + 2 by omega, flatR_len hcs (alg.size - 2) (by omega)]
      omega
    have hdA : d = alg.size := by
      rcases Nat.lt_trichotomy d alg.size with hlt | heq | hgt
      · have := flatR_lt_of_pos (fun k => cr.getD k #[]) hlt (hpos d (by omega))
        rw [← hco] at this
        omega
      · exact heq
      · have := flatR_lt_of_pos (fun k => cr.getD k #[]) hgt (hpos alg.size (by omega))
        rw [← hco] at this
        omega
    subst hdA
    refine ⟨hA, hcs, S0.m, P, hco, hP, hrk, fun h => ?_, fun h hne => ?_⟩
    · rw [ite_eq_left h] at hp1
      rw [hpr]
      exact hp1 rfl
    · rw [ite_eq_right h] at hp2
      rw [hpr]
      exact hp2 hne

end Ranked

namespace Loop

/-- **The coins the loop hands a rep's state are the opened coins**: round `p`'s word `e`, below the round's count. -/
theorem wordsAt_slot (T : ZKMal.CoinLayer.TableData) (x : Msg) (ops : List (Msg × Msg)) (hl : ops.length = refK x)
    {s p e : ℕ} (hp : p < s) (hs : s ≤ ops.length)
    (he : e < (wordsAt x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2) p).size) :
    (wordsAt x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2) p)[e]! =
      Masked.slotAt (x := x) ((ZKMal.CoinLayer.session loopNr T).pad (K := refK x)
        (ops.map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2)) p e := by
  have hpo : p < ops.length := by omega
  have hw : wordsAt x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2) p =
      (ZKMal.CoinLayer.rd ops[p].2).vals (min (loopNr x p) _root_.Flock.Firewall.Frames.COIN_MAX) := by
    unfold wordsAt
    rw [List.getElem?_map, List.getElem?_take_of_lt hp, List.getElem?_eq_getElem hpo]
    rfl
  rw [hw] at he ⊢
  rw [getElem!_pos _ e he]
  unfold Masked.slotAt
  rw [dite_eq_left_of_eq_true (eq_true (show p < refK x by omega))]
  show _ = ((ops.map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2).getD p default).at e
  rw [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_eq_getElem hpo, Option.map_some, Option.getD_some]
  show _ = (ZKMal.CoinLayer.rd ops[p].2).at e
  simp [FlockDeploy.Coin.vals]

theorem wordsAt_size (x : Msg) (ops : List (Msg × Msg)) {s p : ℕ} (hp : p < s) (hs : s ≤ ops.length) :
    (wordsAt x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2) p).size =
      min (loopNr x p) _root_.Flock.Firewall.Frames.COIN_MAX := by
  have hpo : p < ops.length := by omega
  unfold wordsAt
  rw [List.getElem?_map, List.getElem?_take_of_lt hp, List.getElem?_eq_getElem hpo]
  simp [FlockDeploy.Coin.vals]

theorem take_sum_lt {rs : List ℕ} {q i : ℕ} (hq : q < rs.length) (hi : i < rs.getD q 0) :
    (rs.take q).sum + i < rs.sum := by
  rw [← List.sum_take_add_sum_drop rs q, List.drop_eq_getElem_cons hq, List.sum_cons]
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hq, Option.getD_some] at hi
  omega

/-- **The loop asks rep `q`'s round `i` for its plan's count.** -/
theorem loopNr_rep {x : Msg} {s : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hsx : FlockDeploy.setupOf true x = some (s, spec, sts)) {q i : ℕ} (hq : q < spec.repStreams.size)
    (hi : i < (roundsOf spec sts).getD q 0) :
    loopNr x (FlockComplete.repStartZk spec sts q + i) =
      (_root_.Flock.Firewall.Session.repPlan spec sts[q / 2]!).getD i 0 := by
  have hrl : q < (roundsOf spec sts).length := by simpa [roundsOf] using hq
  rw [repStartZk_eq spec sts hq.le, show 1 + ((roundsOf spec sts).take q).sum + i =
    ((roundsOf spec sts).take q).sum + i + 1 by omega]
  have hl := locate_take_sum (roundsOf spec sts) 0 q i hrl hi
  rw [Nat.zero_add] at hl
  exact loopNr_succ hsx hl (take_sum_lt hrl hi)

theorem crAt_getD (f : ℕ → Array F128) (n : ℕ) (b : Array F128) {i : ℕ} (hi : i < n) :
    (((Array.range n).map f).push b).getD i #[] = f i := by
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_push_lt (by simpa using hi)]
  simp

theorem crMap_getD (f : ℕ → Array F128) (n : ℕ) {i : ℕ} (hi : i < n) :
    ((Array.range n).map f).getD i #[] = f i := by
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (by simpa using hi)]
  simp

/-- A rep's start lays out the statement's pads at its schedule's `k0`, and its mask slot. -/
theorem start_eq {st : Stmt} {lay : _root_.Flock.Zk.PadLayout} {mask : MaskSlot}
    (h : _root_.Flock.Firewall.Drive.start st = .ok (lay, mask)) :
    ∃ sch tp, _root_.Flock.Zk.schedule st.m = .ok (sch, tp) ∧
      lay = _root_.Flock.Zk.PadLayout.new st.m st.c.kLog sch.k0 ∧ _root_.Flock.Firewall.Drive.maskSlot st = .ok mask := by
  unfold _root_.Flock.Firewall.Drive.start at h
  obtain ⟨⟨sch, tp⟩, hs, h⟩ := bindV_ok h
  obtain ⟨ms, hm, h⟩ := bindV_ok h
  simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
  obtain ⟨h1, h2⟩ := h
  subst h2
  exact ⟨sch, tp, hs, h1.symm, hm⟩

/-- Names a loop state's coins. -/
theorem repState_args {c c'' : Cfg} {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray}
    {xs : Array (Array F128)} {hi : Array ByteArray} {cr0 cr : Array (Array F128)} {ns : ℕ → ℕ} {n : ℕ}
    {cs : Array F128} {m : Masker} (h : (repState c t rep dom xs hi cr0 cr ns n).answer c'' cs = .ok m) :
    ∃ cr0' cr', cr0' = cr0 ∧ cr' = cr ∧ (repState c t rep dom xs hi cr0' cr' ns n).answer c'' cs = .ok m :=
  ⟨_, _, rfl, rfl, h⟩

end Loop

namespace LeanRelease

open Loop Ranked

/-- **The frames refuse a rank-deficient mask slot**: at a `Live` statement the loop serves, on a live history of
`refK x` openings whose last coins the frame takes, table `j`'s rep `r`'s rank check holds at the opened coins. The
opening at `betaSlot` is the rep's inner round 2, where the frame is past the opening's label and the rank check held
(`repState_rank`) on the rounds' coins, which are the opened ones (`wordsAt_slot`), and, at rep `1`, after rep `0`'s
claim points, whose rounds the loop asks for the same counts (`loopNr_rep`). Both read as the session's
(`frame_points`). -/
theorem rank_unrefused (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (wk : (ofLoop T L₀ real).toSrv.Worker) (x : Msg) (w : Flock.Wit) (hx : Masked.Live x) (hSv : Serves x)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (κ : (ofLoop T L₀ real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hl : ops.length = refK x)
    (hlive : (ofLoop T L₀ real).toSrv.live wk x w ρ κ ((m₀, ρ₀) :: ops) = true)
    (hr : (ofLoop T L₀ real).refusesCoins x ρ ((m₀, ρ₀) :: ops) = false)
    {j r : ℕ} (hj : j < (Masked.stsOf x).size) (hr2 : r < 2) :
    ¬ Masked.RankBad (x := x) ((ZKMal.CoinLayer.session loopNr T).pad (K := refK x)
      (ops.map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2)) j r := by
  intro hbad
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨y⟩ := hx.2 j hj
  obtain ⟨-, -, hspec, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hK := Masked.refK_eq hsx
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
  obtain ⟨N, hN⟩ : ∃ N, N = FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * j + r) / 2]! +
      (y.sch.k0 + 7 + 11 * (y.sch.levels.size - 1)) := ⟨_, rfl⟩
  have hrR : ∀ q, q / 2 = (2 * j + r) / 2 → FlockDeploy.repRounds true spec (Masked.stsOf x)[q / 2]! = N + 4 := by
    intro q hq'
    rw [hq']
    have e1 : (Masked.stsOf x)[(2 * j + r) / 2]!.m + (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog - 6 =
        (Masked.stsOf x)[(2 * j + r) / 2]!.m + ((Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog - 6) := by omega
    rw [repRounds_true]
    simp only [hz]
    rw [hN]
    unfold FlockComplete.piopRoundsZk
    rw [e1]
  have hrq : ∀ q, q / 2 = (2 * j + r) / 2 → q < spec.repStreams.size →
      (roundsOf spec (Masked.stsOf x)).getD q 0 = N + 4 := by
    intro q hq' hq
    simp only [roundsOf, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hq, Option.map_some,
      Option.getD_some]
    exact hrR q hq'
  have hrsl : (roundsOf spec (Masked.stsOf x)).length = spec.repStreams.size := by simp [roundsOf]
  have hend : FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) +
      FlockDeploy.repRounds true spec (Masked.stsOf x)[(2 * j + r) / 2]! ≤ refK x := by
    rw [← FlockComplete.ZkTables.repStart_succ, hK, ← FlockComplete.ZkTables.repStart_last]
    exact FlockComplete.ZkTables.repStart_mono spec (Masked.stsOf x) (by omega)
  rw [hrR _ rfl] at hend
  have hs1 : 1 ≤ FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) := by
    unfold FlockComplete.repStartZk; omega
  obtain ⟨s, hs⟩ : ∃ s, s = FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) + N + 2 := ⟨_, rfl⟩
  have hb : s < refK x := by omega
  have hlast : (ofLoop T L₀ real).last x = refK x + 1 := loop_last loopNr T (withFrames T L₀ real) x
  obtain ⟨-, m, ha⟩ := coins_taken_of_live (ofLoop T L₀ real) wk x w ρ κ ((m₀, ρ₀) :: ops) hlive
    (by rw [hlast]; simp [hl]) hr (i := s + 2) (by omega) (by simp only [List.length_cons]; omega)
  obtain ⟨⟨dm, da⟩, hd⟩ : ∃ d, ops[s]? = some d := ⟨ops[s]'(by omega), List.getElem?_eq_getElem _⟩
  have htk : ops.take (s + 1) = ops.take s ++ [(dm, da)] := by rw [List.take_add_one, hd]; rfl
  have hh : ((m₀, ρ₀) :: ops).take (s + 2) = (m₀, ρ₀) :: (ops.take s ++ [(dm, da)]) := by
    rw [List.take_succ_cons, htk]
  have hts : (ops.take s).length = s := by rw [List.length_take]; omega
  rw [hh] at ha
  change (progOf x ((ops.take s ++ [(dm, da)]).map fun p => ZKMal.CoinLayer.rd p.2)).answer _
    (coinsAt x ((m₀, ρ₀) :: (ops.take s ++ [(dm, da)]))) = .ok m at ha
  rw [List.map_append, List.map_cons, List.map_nil] at ha
  have hloc : ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (s - 1) = (2 * j + r, N + 2) := by
    have e := locate_take_sum (roundsOf spec (Masked.stsOf x)) 0 (2 * j + r) (N + 2) (by rw [hrsl]; exact hq2)
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
  obtain ⟨alg, sched, hal, hsc⟩ := repState_ok ha
  obtain ⟨hts', ⟨tp, htp⟩, hxs⟩ := Count.placeholder_tab hph
  rw [hz, Except.ok.injEq, Prod.mk.injEq] at htp
  obtain ⟨htsch, -⟩ := htp
  obtain ⟨hm36, hk0, hn, hq⟩ := Count.sched_facts (htsch ▸ hz : _root_.Flock.Zk.schedule _ = .ok (t.sch, y.tPad))
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
    have e := congrArg List.length (Count.rounds_counts hmb hal)
    rw [Count.piop_counts, hts', List.length_map, Array.length_toList] at e
    exact e
  have hplan := congrArg List.length (Count.repPlan_list (spec := spec) (htsch ▸ hz :
    _root_.Flock.Zk.schedule _ = .ok (t.sch, y.tPad)))
  rw [Array.length_toList, repPlan_size] at hplan
  simp only [repRounds_true, hz, List.length_append] at hplan
  have h4 : ([1, _root_.Flock.Firewall.Session.INNER_CONS, 1, _root_.Flock.Zk.Q].filter (· != 0)).length = 4 := by
    decide
  have hFO : alg.size + (Count.openList (2 + spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size)
      t.sch).length = N := by
    rw [h4] at hplan
    have e1 : (Masked.stsOf x)[(2 * j + r) / 2]!.m + (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog - 6 =
        (Masked.stsOf x)[(2 * j + r) / 2]!.m + ((Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog - 6) := by omega
    rw [hA, hN]
    unfold FlockComplete.piopRoundsZk
    rw [e1]
    exact (Nat.add_right_cancel hplan).symm
  have hx2 : xs.size + 2 = 2 + spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size := by omega
  have hS := Count.schedSoFar_size t _ dom xs hi _ hal hsc (hdm ▸ hdom') hdg (placeholder_cap hph) hmb hxb hk0 hn hq
    (by
      rw [hx2, hFO, hCR]
      simp only [Array.size_pop, Array.size_push, Array.size_map, Array.size_range]
      omega)
  rw [hx2, hFO] at hS
  rw [show N + 2 + 1 = sched.size + 2 + 1 by omega] at ha
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
  obtain ⟨hA', hcs, S, P, hco, hP, hrk, hp0, hp1⟩ := repState_rank hcc hal hsc
    (by rw [htm, hstj]; exact hm14) (by rw [htk, hstj]; exact y.wf.1.hk6) hmb hdg (placeholder_cap hph)
    (hdm ▸ hdom') hmask hlay ha
  rw [htm, htk] at hA' hlay
  rw [htm] at hcs
  have hm14' : 14 ≤ (Masked.stsOf x)[j]!.m := by rw [hstj]; exact hm14
  have hk6' : 6 ≤ (Masked.stsOf x)[j]!.c.kLog := by rw [hstj]; exact y.wf.1.hk6
  have hAN : alg.size ≤ N := by omega
  have hW : ∀ i < N + 2, CR.getD i #[] =
      wordsAt x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2)
        (FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) + i) := by
    intro i hi'
    rw [hCR, crAt_getD _ _ _ hi']
    generalize hR0 : FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) = R0
    rw [show s - 1 - (N + 2) + i + 1 = R0 + i by omega]
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
        rw [show s - 1 - (N + 2) - (N + 4) + i + 1 = R0 + i by omega]
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
