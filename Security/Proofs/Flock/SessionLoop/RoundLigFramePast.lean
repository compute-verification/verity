module

public import Proofs.Flock.SessionLoop.RoundLigFrameBind
public import Proofs.Flock.SessionLoop.RoundLigFrameShape
public import Proofs.Flock.Claims.ZK.FrameSched

/-!
# A rep's schedule past level 0's close

Past level 0's closing round `C` (the rep's algebraic phase, the ring switch's and `γ`'s rounds, then `k0 + 4`), each
round a rep's schedule holds, up to its next, is `ligeritoPZk`'s body there (`Bind.sched_lig`), so it is the shape
`LigShape.ligeritoPZk_logQ` gives Ligerito's bodies from the close on: the closing label at `C`, then good frames
(`LigShape.Good`), then a squeeze of the round's coin count.

* `LigOff.schedSoFar_pairs`, `LigOff.feedR_counts`: past the algebraic phase a rep's schedule is the rounds its opening,
  fed, sent and asks for next, each with its count, which its bytes' last eight hold (`Fed.sized_openStep`).
* **`LigOff.sched_past`**: fed fewer coins than the opening asks for, each round of the schedule from `C` to the rounds
  it was fed is `PastRound`'s.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer.LigOff

open _root_.Flock (F128)
open _root_.Flock.Transcript (header)
open _root_.Flock.Firewall.Drive (Step openStep)
open FlockSoundness.Refine (Frame encs)
open FlockComplete.LigLock.Fed (feedR feed_eq Sized CoinAt)
open Loop

/-- **A scheduled round past level 0's close at `C`**: at round `p`, its bytes are the closing label if `p = C`, good
frames, then a squeeze of its count `r.2` of coins, `r.2 ≠ 0` and below `2 ^ 64`, as the frame takes it. -/
def PastRound (C p : ℕ) (r : ByteArray × ℕ) : Prop :=
  ∃ (fs : List Frame) (k : UInt8),
    r.1.data.toList = encs (FlockComplete.LigShape.preAt C p ++ fs) ++ (Frame.sq k r.2).enc ∧
    (∀ f ∈ fs, FlockComplete.LigShape.Good f) ∧ (k = 2 ∨ (k = 1 ∧ r.2 = 1)) ∧ r.2 ≠ 0 ∧ r.2 < 2 ^ 64

/-- The round a step program asks for next, with its count, if it does. -/
def pendP {α : Type} : Step α → List (ByteArray × ℕ)
  | .ask c n _ => [(c, n)]
  | _ => []

/-- **A feeding's rounds hold their counts**: each round a program that checks its coins' counts sent, or asks for
next, holds its count in its bytes' last eight. -/
theorem feedR_counts {α : Type} : ∀ (s : Step α) (cs : List (Array F128)), Sized s →
    ∀ q ∈ (feedR s cs).2.1 ++ pendP (feedR s cs).1, q.2 = (_root_.Flock.u64OfLE q.1 (q.1.size - 8)).toNat
  | .done a, cs, _, q, hq => by
    have e : (feedR (Step.done a) cs).2.1 ++ pendP (feedR (Step.done a) cs).1 = [] := by cases cs <;> rfl
    rw [e] at hq
    simp at hq
  | .fail e, cs, _, q, hq => by
    have e : (feedR (Step.fail e : Step α) cs).2.1 ++ pendP (feedR (Step.fail e : Step α) cs).1 = [] := by
      cases cs <;> rfl
    rw [e] at hq
    simp at hq
  | .ask c n k, [], h, q, hq => by
    have e : (feedR (Step.ask c n k) []).2.1 ++ pendP (feedR (Step.ask c n k) []).1 = [(c, n)] := rfl
    rw [e, List.mem_singleton] at hq
    subst hq
    exact h.1
  | .ask c n k, z :: rest, h, q, hq => by
    have e : (feedR (Step.ask c n k) (z :: rest)).2.1 ++ pendP (feedR (Step.ask c n k) (z :: rest)).1 =
        (c, n) :: ((feedR (k z) rest).2.1 ++ pendP (feedR (k z) rest).1) := rfl
    rw [e] at hq
    rcases List.mem_cons.mp hq with rfl | hq
    · exact h.1
    · exact feedR_counts (k z) rest (h.2.2 z) q hq

/-- **A rep's schedule past its algebraic phase, with its counts**: the rounds its opening, fed the rest of the coins,
sent and asks for next. -/
theorem schedSoFar_pairs {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray}
    {xs : Array (Array F128)} {hi : Array ByteArray} {cr : Array (Array F128)} {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr = .ok sched) (hA : alg.size ≤ cr.size) :
    sched.toList.drop alg.size =
      (feedR (openStep t rep dom xs hi (cr.extract 0 alg.size)) (cr.extract alg.size cr.size).toList).2.1 ++
        pendP (feedR (openStep t rep dom xs hi (cr.extract 0 alg.size)) (cr.extract alg.size cr.size).toList).1 := by
  unfold schedSoFar at hs
  rw [hal] at hs
  simp only [bind, Except.bind] at hs
  split at hs
  · omega
  · rw [feed_eq] at hs
    rcases hF : feedR (openStep t rep dom xs hi (cr.extract 0 alg.size)) (cr.extract alg.size cr.size).toList with
      ⟨s', sent, rest⟩
    rw [hF] at hs
    cases s' with
    | ask c n k =>
      simp only [pure, Except.pure, Except.ok.injEq] at hs
      subst hs
      try rw [hF]
      rw [Array.toList_append, List.drop_left' (by simp)]
      simp [pendP]
    | done a =>
      simp only [pure, Except.pure, Except.ok.injEq] at hs
      subst hs
      try rw [hF]
      rw [Array.toList_append, List.drop_left' (by simp)]
      simp [pendP]
    | fail e => cases hs

set_option maxHeartbeats 1000000 in
/-- **A rep's schedule past level 0's close**: fed fewer coins than the opening asks for, each round from the close
`C = alg.size + (xs.size + 2) + 1 + k0 + 4` to the rounds fed, the next included, is `PastRound`'s. -/
theorem sched_past {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} {cr : Array (Array F128)} {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr = .ok sched) (hdom : dom.size < 2 ^ 64)
    (hdg : t.st.digest.size < 2 ^ 64) (hcn : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64)
    (hm : t.st.m < 2 ^ 64) (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (hx : xs.size + 2 < 2 ^ 64)
    (hk : 1 ≤ t.sch.k0) (hn : t.sch.n < 2 ^ 62) (hq : ∀ i, (t.sch.levels.getD i default).q < 2 ^ 63)
    (hm14 : 14 ≤ t.st.m) (hn0 : t.sch.n ≠ 0) (htp : 2 * t.tPad < 2 ^ 64)
    (hd : ∀ i, 64 * 2 ^ (t.sch.levels.getD i default).d < 2 ^ 64)
    (hc2 : cr.size < alg.size + (Count.openList (xs.size + 2) t.sch).length) {p : ℕ}
    (hp1 : alg.size + (xs.size + 2) + 1 + t.sch.k0 + 4 ≤ p) (hp2 : p ≤ cr.size) :
    p < sched.size ∧
      PastRound (alg.size + (xs.size + 2) + 1 + t.sch.k0 + 4) p (sched.getD p (.empty, 0)) := by
  have hA : alg.size ≤ cr.size := by omega
  obtain ⟨l, hls⟩ : ∃ l : Array Claims.Msg, l.size = alg.size := ⟨Array.replicate alg.size [], Array.size_replicate⟩
  obtain ⟨coin, hc⟩ : ∃ coin : ℕ → ℕ → F128, CoinAt coin l.size (cr.extract alg.size cr.size).toList :=
    ⟨fun j i => ((cr.extract alg.size cr.size).toList.getD (j - l.size) #[])[i]!, fun j hj => by
      funext i
      simp only [Nat.add_sub_cancel_left, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj,
        Option.getD_some]⟩
  obtain ⟨hlt, T, sws, tr₂, l₂, L, hl₂, -, hrun, hL, -, -⟩ :=
    Bind.sched_lig hal hs hdom hdg hcn hm hmk hx hk hn hq hm14 (by omega) hc2 coin l hc (n := cr.size)
      (by rw [hls]; omega)
  have hn' : t.sch.n + 1 < 2 ^ 64 := lt_of_lt_of_le (Nat.succ_lt_succ hn) (by norm_num)
  have hq' : ∀ i, (t.sch.levels.getD i default).q < 2 ^ 64 := fun i => lt_of_lt_of_le (hq i) (by norm_num)
  have hQ := FlockComplete.LigShape.ligeritoPZk_logQ coin cr.size t.sch T t.cm0.toC t.g
    (FlockComplete.basisVals sws (t.st.m - 7)) (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0)
    (Array.replicate (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0).len 0) rep t.tPad t.extra t.lanePad
    (fun l => _root_.Flock.Firewall.Finish.saltAt (hi.getD (l - 1) ByteArray.empty)) (fun _ _ => 0) tr₂ l₂
    hk hn0 hn' hq' htp hd (by rw [hl₂, hls]; omega)
  rw [hrun, hl₂, hls] at hQ
  have hps : p < sched.size := by omega
  have hsz : L.size = sched.size := by
    rw [hL, Array.size_append, hls]
    simp only [List.size_toArray, List.length_map, List.length_drop, Array.length_toList]
    omega
  have hpL : p < L.size := by rw [hsz]; exact hps
  obtain ⟨fs, k, n', hb, hG, hkn, hn0', hn64⟩ := hQ p hpL hp1
  have hlp : l.size ≤ p := by rw [hls]; omega
  have hLp : L[p]? = some (Claims.toMsg (sched.getD p (.empty, 0)).1) := by
    rw [hL, FlockComplete.LigLock.Fed.log_at hlp, hls, List.getElem?_drop,
      show alg.size + (p - alg.size) = p by omega, List.getElem?_map, Array.getElem?_toList,
      Array.getElem?_eq_getElem hps, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hps]
    rfl
  rw [Array.getElem?_eq_getElem hpL, Option.some.injEq] at hLp
  have hr1 : (sched.getD p (.empty, 0)).1.data.toList =
      encs (FlockComplete.LigShape.preAt (alg.size + (xs.size + 2) + 1 + t.sch.k0 + 4) p ++ fs) ++
        (Frame.sq k n').enc :=
    hLp.symm.trans hb
  have hcnt : (sched.getD p (.empty, 0)).2 =
      (_root_.Flock.u64OfLE (sched.getD p (.empty, 0)).1 ((sched.getD p (.empty, 0)).1.size - 8)).toNat := by
    have hmem : sched.getD p (.empty, 0) ∈ sched.toList.drop alg.size := by
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hps, Option.getD_some]
      have h1 : (sched.toList.drop alg.size)[p - alg.size]? = some sched[p] := by
        rw [List.getElem?_drop, show alg.size + (p - alg.size) = p by omega, Array.getElem?_toList,
          Array.getElem?_eq_getElem hps]
      exact List.mem_of_getElem? h1
    rw [schedSoFar_pairs hal hs hA] at hmem
    exact feedR_counts _ _ (FlockComplete.LigLock.Fed.sized_openStep t rep dom xs hi _ hmk) _ hmem
  have hbytes : (sched.getD p (.empty, 0)).1 =
      ByteArray.mk (encs (FlockComplete.LigShape.preAt (alg.size + (xs.size + 2) + 1 + t.sch.k0 + 4) p ++
        fs)).toArray ++ header 4 k n' := by
    apply ByteArray.ext
    apply Array.ext'
    rw [hr1, Claims.ZKServer.data_append', Claims.ZKServer.sq_enc]
  have hn2 : (sched.getD p (.empty, 0)).2 = n' := by
    rw [hcnt, hbytes]
    exact Sched.count_header _ k n' hn64
  refine ⟨hps, ?_⟩
  unfold PastRound
  refine ⟨fs, k, ?_, hG, ?_, ?_, ?_⟩
  · rw [hr1, hn2]
  · rw [hn2]
    exact hkn
  · rw [hn2]
    exact hn0'
  · rw [hn2]
    exact hn64

end Claims.ZKServer.LigOff
