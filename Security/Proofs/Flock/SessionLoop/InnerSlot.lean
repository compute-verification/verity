module

public import Proofs.Flock.Claims.ZK.ExePlan
public import Proofs.Flock.Claims.ZK.FramePlan
public import Proofs.Flock.Claims.ZK.FrameZK
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Complete.ZkTables.Scope
import Proofs.Flock.Claims.Complete.ZkTables.Table
import Proofs.Flock.Claims.DeployK
import Proofs.Flock.Claims.ZK.FrameBeta
import Proofs.Flock.Claims.ZK.FrameRankFault

/-!
# Where the inner proof's rounds sit in the session

`Loop.InnerAt` places a history at a rep's rounds past the end of the schedule so far on its own table. There the
schedule is the whole of the rep's algebraic phase and opening, so the history's round is the inner proof's round
`i - sched.size`, below `4`, and the session's message there is `Masked.repSlot … (Masked.innerOff (i - sched.size))`:

* `Count.feed_short`: fed fewer coins than it asks for, a step program that ends with no asks left has not ended.
* `Count.schedSoFar_full`: a schedule so far that holds no round past its coins is the algebraic phase's rounds and all
  of the opening's (`schedSoFar_size`, whose coin count it supplies).
* `Loop.locate_inv`: `CoinLayer.locate` within the rounds' sum places a round at its stream's start plus its index, below
  the stream's length.
* `Loop.getD_append_at`: past two lists of `N` entries, entry `N + e` is the third list's `e`-th.
* **`Loop.innerAt_slot`**: at a `Live` statement the loop serves, a history below `refK x` at `InnerAt` on tables `real`
  answers as the loop's own (`Count.RealTabs`) is table `k / 2`'s rep `k % 2`'s inner round `i - sched.size < 4`, at
  `repSlot (2j + r) (innerOff (i - sched.size))`, within the rep's rounds, where the loop asks the inner proof's rounds
  for `1`, `INNER_CONS`, `1` and `Q` coins.
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)
open _root_.Flock.Firewall.Finish (Tab)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoopLink`'s session types, as in `TakenVals`

namespace Count

open _root_.Flock.Firewall.Drive (Step)

/-- **Fed fewer coins than it asks for, a step program that ends with no asks left has not ended.** -/
theorem feed_short {α : Type} : ∀ (s : Step α) (cs : List (Array F128)) (rs : Array (ByteArray × ℕ)) (ns : List ℕ),
    SP (fun _ ns => ns = []) s ns → cs.length < ns.length → ∀ a rs', s.feed cs rs ≠ (.done a, rs')
  | .done _, _, _, ns, h, hl => by
    have h0 : ns = [] := h
    subst h0
    simp at hl
  | .fail e, cs, rs, _, _, _ => by
    intro a rs' h'
    have e' : (Step.fail e : Step α).feed cs rs = (.fail e, rs) := by cases cs <;> rfl
    rw [e'] at h'
    simp at h'
  | .ask c n k, [], rs, _, _, _ => by
    intro a rs' h'
    have e' : (Step.ask c n k).feed [] rs = (.ask c n k, rs) := rfl
    rw [e'] at h'
    simp at h'
  | .ask c n k, x :: rest, rs, ns, ⟨ns', e, hk⟩, hl => by
    intro a rs' h'
    subst e
    have e' : (Step.ask c n k).feed (x :: rest) rs = (k x).feed rest (rs.push (c, n)) := rfl
    rw [e'] at h'
    exact feed_short (k x) rest (rs.push (c, n)) ns' (hk x) (by simp only [List.length_cons] at hl; omega) a rs' h'

/-- **A schedule so far that holds no round past its coins is complete**: the algebraic phase's rounds and all of the
opening's. Short of the opening's coins, it would hold the round the opening asks for next. -/
theorem schedSoFar_full (t : Tab) (rep : ℕ) (dom : ByteArray) (xs : Array (Array F128)) (hi : Array ByteArray)
    (cr : Array (Array F128)) {alg sched : Array (ByteArray × ℕ)}
    (ha : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (h : Loop.schedSoFar t rep dom xs hi cr = .ok sched) (hs : sched.size ≤ cr.size) (hdom : dom.size < 2 ^ 64)
    (hdg : t.st.digest.size < 2 ^ 64) (hcn : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64)
    (hm : t.st.m < 2 ^ 64) (hx : xs.size + 2 < 2 ^ 64) (hk : 1 ≤ t.sch.k0) (hn : t.sch.n < 2 ^ 62)
    (hq : ∀ i, (t.sch.levels.getD i default).q < 2 ^ 63) :
    sched.size = alg.size + (openList (xs.size + 2) t.sch).length := by
  refine schedSoFar_size t rep dom xs hi cr ha h hdom hdg hcn hm hx hk hn hq ?_
  have h' := h
  unfold Loop.schedSoFar at h'
  rw [ha] at h'
  simp only [bind, Except.bind] at h'
  split at h'
  · simp only [pure, Except.pure, Except.ok.injEq] at h'
    subst h'
    omega
  · rename_i hge
    have hlen : (cr.extract alg.size cr.size).toList.length = cr.size - alg.size := by
      rw [Array.length_toList, Array.size_extract]
      omega
    rcases hfe : (_root_.Flock.Firewall.Drive.openStep t rep dom xs hi (cr.extract 0 alg.size)).feed
      (cr.extract alg.size cr.size).toList #[] with ⟨s', rs⟩
    rw [hfe] at h'
    cases s' with
    | ask c n k =>
      simp only [pure, Except.pure, Except.ok.injEq] at h'
      subst h'
      have e := Loop.feed_ask_size _ _ _ hfe
      rw [Array.size_append, Array.size_push, e, hlen, Array.size_empty] at hs
      omega
    | done a =>
      exact Nat.le_of_not_lt fun hlt => feed_short _ _ #[] _
        (openStep_end t rep dom xs hi (cr.extract 0 alg.size) hdom hdg hcn hm hx hk hn hq)
        (by rw [hlen]; omega) a rs hfe
    | fail e => cases h'

end Count

namespace Loop

/-- **`locate` within the rounds' sum**: round `q` is stream `(locate rs k q).1`'s round `(locate rs k q).2`, at the
stream's start `Σ` of the streams before it, below the stream's length. -/
theorem locate_inv : ∀ (rs : List ℕ) (k q : ℕ), q < rs.sum →
    (rs.take ((CoinLayer.locate rs k q).1 - k)).sum + (CoinLayer.locate rs k q).2 = q ∧
      (CoinLayer.locate rs k q).2 < rs.getD ((CoinLayer.locate rs k q).1 - k) 0
  | [], _, _, h => by simp at h
  | c :: cs, k, q, h => by
    by_cases hc : q < c
    · rw [locate_cons_lt hc]
      simp [hc]
    · rw [locate_cons_ge hc]
      have ih := locate_inv cs (k + 1) (q - c) (by simp only [List.sum_cons] at h; omega)
      have hge := locate_fst_ge cs (k + 1) (q - c)
      obtain ⟨d, hd⟩ : ∃ d, (CoinLayer.locate cs (k + 1) (q - c)).1 - k = d + 1 :=
        ⟨(CoinLayer.locate cs (k + 1) (q - c)).1 - k - 1, by omega⟩
      rw [show (CoinLayer.locate cs (k + 1) (q - c)).1 - (k + 1) = d by omega] at ih
      rw [hd, List.take_succ_cons, List.sum_cons, List.getD_cons_succ]
      omega

/-- **Past two lists of `N` entries, entry `N + e` is the third list's `e`-th.** -/
theorem getD_append_at {A B C : List ℕ} {N e : ℕ} (h : A.length + B.length = N) :
    (A ++ B ++ C)[N + e]?.getD 0 = C.getD e 0 := by
  have hl : (A ++ B).length ≤ N + e := by
    rw [List.length_append]
    omega
  rw [List.getElem?_append_right hl, List.length_append, show N + e - (A.length + B.length) = e by omega,
    List.getD_eq_getElem?_getD]

/-- **The inner proof's rounds sit past the rep's whole schedule**: at a `Live` statement the loop serves, a history of
`q + 1 < refK x` decoded coins at the inner proof's rounds (`InnerAt`'s components: rep stream `k`'s round `i`, at or
past the end of its schedule so far `rsched` on the table `real` answers as the loop's own) is table `k / 2`'s rep
`k % 2`'s inner round `i - rsched.size`, below `4`, and the session's round there is
`repSlot (2j + r) (innerOff (i - rsched.size))`, within the rep's rounds (`repBase`), and at most `q`. The schedule so
far is complete (`Count.schedSoFar_full`), the rep's algebraic phase and opening (`Count.repPlan_list`), so it ends at
the rep's `piopRoundsZk + k₀ + 7 + 11(R − 1)`, past which the rep's plan, and so the loop (`loopNr_rep`), asks the inner
proof's rounds for `1`, `INNER_CONS`, `1` and `Q` coins. -/
theorem innerAt_slot {real : ℕ → V (Tab × Array ByteArray)} {x : Msg} (hx : Masked.Live x) (hSv : Serves x)
    {cs : List FlockDeploy.Coin} {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} {q k i : ℕ}
    {tr : Tab} {hr : Array ByteArray} {rsched : Array (ByteArray × ℕ)}
    (hs : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : cs.length = q + 1) (hK : cs.length < refK x)
    (hloc : CoinLayer.locate (roundsOf spec sts) 0 q = (k, i)) (hR : Count.RealTabs sts real)
    (hreal : real k = .ok (tr, hr))
    (hrs : schedSoFar tr (k % 2) (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8
      (_root_.Flock.Firewall.Drive.extras sts[k / 2]!
        (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]! (wordsAt x cs 0)))
      hr ((Array.range i).map fun j => wordsAt x cs (q - i + j + 1)) = .ok rsched)
    (hle : rsched.size ≤ i) :
    ∃ (j : Fin (Masked.jOf x)) (r : Fin 2), k = 2 * j.val + r.val ∧ i - rsched.size < 4 ∧
      q + 1 = Masked.repSlot x (2 * j.val + r.val) (Masked.innerOff (i - rsched.size)) ∧
      Masked.repBase x (2 * j.val + r.val) ≤ q + 1 ∧ q + 1 < Masked.repBase x (2 * j.val + r.val + 1) ∧
      0 < i ∧ i ≤ q ∧ ∀ e < 4, loopNr x (Masked.repSlot x (2 * j.val + r.val) (Masked.innerOff e)) =
        [1, _root_.Flock.Firewall.Session.INNER_CONS, 1, _root_.Flock.Zk.Q].getD e 0 := by
  have hsts : Masked.stsOf x = sts := by
    unfold Masked.stsOf
    rw [hs]
  subst hsts
  obtain ⟨-, -, hspec, -⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hs
  have hrsz : spec.repStreams.size = 2 * (Masked.stsOf x).size := by rw [FlockDeploy.repStreams_size, hspec]
  have hrsl : (roundsOf spec (Masked.stsOf x)).length = spec.repStreams.size := by simp [roundsOf]
  have hsum : (roundsOf spec (Masked.stsOf x)).sum + 1 = refK x := by
    rw [Masked.refK_eq hs, ← FlockComplete.ZkTables.repStart_last, ← hrsz]
    unfold FlockComplete.repStartZk roundsOf
    omega
  have hinv := locate_inv (roundsOf spec (Masked.stsOf x)) 0 q (by omega)
  rw [hloc] at hinv
  simp only [Nat.sub_zero] at hinv
  obtain ⟨hqs, hil⟩ := hinv
  have hk : k < spec.repStreams.size := by
    have := locate_lt (roundsOf spec (Masked.stsOf x)) 0 q (by omega)
    rw [hloc] at this
    simp only [hrsl, Nat.zero_add] at this
    omega
  have hj : k / 2 < (Masked.stsOf x).size := by omega
  obtain ⟨y⟩ := hx.2 (k / 2) hj
  have hst : (Masked.stsOf x)[k / 2]! = (Masked.stsOf x)[k / 2] := getElem!_pos _ (k / 2) hj
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[k / 2]!.m = .ok (y.sch, y.tPad) := by
    rw [hst]
    exact y.hz
  have hk6 : 6 ≤ (Masked.stsOf x)[k / 2]!.c.kLog := by
    rw [hst]
    exact y.wf.1.hk6
  obtain ⟨N, hN⟩ : ∃ N, N = FlockComplete.piopRoundsZk spec (Masked.stsOf x)[k / 2]! +
      (y.sch.k0 + 7 + 11 * (y.sch.levels.size - 1)) := ⟨_, rfl⟩
  have hrq : (roundsOf spec (Masked.stsOf x)).getD k 0 = N + 4 := by
    simp only [roundsOf, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hk, Option.map_some,
      Option.getD_some]
    have e1 : (Masked.stsOf x)[k / 2]!.m + (Masked.stsOf x)[k / 2]!.c.kLog - 6 =
        (Masked.stsOf x)[k / 2]!.m + ((Masked.stsOf x)[k / 2]!.c.kLog - 6) := by omega
    rw [repRounds_true]
    simp only [hz]
    rw [hN]
    unfold FlockComplete.piopRoundsZk
    rw [e1]
  -- the rep's own table, `tabOfLevel0` at a sampled level 0, and its bounds
  obtain ⟨g, d, l, hl, htr⟩ := hR k tr hr hreal
  obtain ⟨hts, hsch⟩ := Count.tabOfLevel0_sched htr
  have htsch : tr.sch = y.sch := by
    have e : _root_.Flock.Zk.schedule (Masked.stsOf x)[k / 2]!.m = .ok (tr.sch, tr.tPad) := hsch
    rw [hz, Except.ok.injEq, Prod.mk.injEq] at e
    exact e.1.symm
  have hcn : (_root_.Flock.Firewall.Finish.flat tr.cm0.cap).size < 2 ^ 64 := by
    obtain ⟨s, tp, hs', -⟩ := Count.ofDraws_gsize hl
    obtain ⟨-, -, ⟨s1, hs1, hlv⟩, -⟩ := _root_.Flock.Firewall.Level0.ofDraws_ok hl
    rw [hs'] at hs1
    obtain ⟨rfl, -⟩ := Prod.mk.inj (Except.ok.inj hs1)
    have hd : 64 * 2 ^ l.lv.d < 2 ^ 64 := by
      rw [hlv]
      exact (Count.sched_bounds hs').2 0
    have hA := Sched.tabOfLevel0_alike (_root_.Flock.Setup.ofCircuit (Masked.stsOf x)[k / 2]!) hl hl rfl hd
    rw [htr] at hA
    have hA' : Sched.TabAlike tr tr := hA
    exact hA'.2.2.2.2.2
  obtain ⟨hm36, hk0, hn, hqb⟩ := Count.sched_facts hz
  obtain ⟨⟨hmp, hmp0, hpc⟩, hdom, hxr, -⟩ := sizesOk_at (sizesOk_of_serves hSv s0 spec _ hs) hk
  have hdg : tr.st.digest.size < 2 ^ 64 := by
    rw [hts]
    simp only [_root_.Flock.Setup.ofCircuit, ByteArray.size_extract]
    omega
  have hmb : tr.st.m < 2 ^ 64 := by
    rw [hts]
    exact Nat.lt_trans hm36 (by decide)
  have hls : (wordsAt x cs 0).size = min (loopNr x 0) _root_.Flock.Firewall.Frames.COIN_MAX := by
    cases cs with
    | nil => simp at hq
    | cons p rest => simp [wordsAt, FlockDeploy.Coin.vals]
  have hl0 : loopNr x 0 = spec.points * spec.mPts := by
    show (reqOf x 0).2 = _
    rw [reqOf_eq hs]
    rfl
  have hpts : (wordsAt x cs 0).size / (Masked.stsOf x)[k / 2]!.mPts = spec.points := by
    rw [hls, hl0, Nat.min_eq_left hpc, hmp, Nat.mul_div_cancel _ hmp0]
  have hxn : (_root_.Flock.Firewall.Drive.extras (Masked.stsOf x)[k / 2]!
      (_root_.Flock.Firewall.Drive.linkPoints (Masked.stsOf x)[k / 2]! (wordsAt x cs 0))).size =
      spec.points * (Masked.stsOf x)[k / 2]!.regions.size := by
    rw [Count.extras_size, Count.linkPoints_size, hpts, Nat.mul_comm]
  have hxb : (_root_.Flock.Firewall.Drive.extras (Masked.stsOf x)[k / 2]!
      (_root_.Flock.Firewall.Drive.linkPoints (Masked.stsOf x)[k / 2]! (wordsAt x cs 0))).size + 2 < 2 ^ 64 := by
    rw [extras_size]
    have := Nat.mul_le_mul_left (Masked.stsOf x)[k / 2]!.regions.size
      ((linkPoints_size_le (Masked.stsOf x)[k / 2]! (wordsAt x cs 0)).trans (hls.trans_le (Nat.min_le_right _ _)))
    omega
  -- the schedule so far is complete: the algebraic phase and the opening, `N` rounds
  obtain ⟨alg, hal⟩ := schedSoFar_ok_rounds hrs
  have hA : alg.size = ((_root_.Flock.Firewall.Session.replayCoins (Masked.stsOf x)[k / 2]!.m
      (Masked.stsOf x)[k / 2]!.c.kLog).toList.filter (· != 0)).length := by
    have e := congrArg List.length (Count.rounds_counts hmb hal)
    rw [Count.piop_counts, hts, List.length_map, Array.length_toList] at e
    exact e
  have hz' : _root_.Flock.Zk.schedule (Masked.stsOf x)[k / 2]!.m = .ok (tr.sch, y.tPad) := by
    rw [htsch]
    exact hz
  have hplan := congrArg List.length (Count.repPlan_list (spec := spec) hz')
  rw [Array.length_toList, repPlan_size] at hplan
  simp only [repRounds_true, hz, List.length_append] at hplan
  have h4 : ([1, _root_.Flock.Firewall.Session.INNER_CONS, 1, _root_.Flock.Zk.Q].filter (· != 0)).length = 4 := by
    decide
  have hFO : alg.size + (Count.openList (2 + spec.points * (Masked.stsOf x)[k / 2]!.regions.size)
      tr.sch).length = N := by
    rw [h4] at hplan
    have e1 : (Masked.stsOf x)[k / 2]!.m + (Masked.stsOf x)[k / 2]!.c.kLog - 6 =
        (Masked.stsOf x)[k / 2]!.m + ((Masked.stsOf x)[k / 2]!.c.kLog - 6) := by omega
    rw [hA, hN]
    unfold FlockComplete.piopRoundsZk
    rw [e1]
    exact (Nat.add_right_cancel hplan).symm
  have hS := Count.schedSoFar_full tr _ _ _ hr _ hal hrs
    (by simp only [Array.size_map, Array.size_range]; exact hle) hdom hdg hcn hmb hxb (by rw [htsch]; exact hk0)
    (by rw [htsch]; exact hn) (by rw [htsch]; exact hqb)
  rw [hxn, show spec.points * (Masked.stsOf x)[k / 2]!.regions.size + 2 =
    2 + spec.points * (Masked.stsOf x)[k / 2]!.regions.size by omega, hFO] at hS
  have hj' : k / 2 < Masked.jOf x := by
    unfold Masked.jOf
    rw [if_pos hx]
    exact hj
  have hrR : FlockDeploy.repRounds true spec (Masked.stsOf x)[k / 2]! = N + 4 := by
    have e := hrq
    simp only [roundsOf, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hk, Option.map_some,
      Option.getD_some] at e
    exact e
  rw [hrq] at hil
  refine ⟨⟨k / 2, hj'⟩, ⟨k % 2, Nat.mod_lt _ (by norm_num)⟩, (Nat.div_add_mod k 2).symm, by omega, ?_, ?_, ?_,
    by omega, by omega, ?_⟩
  · show q + 1 = Masked.repSlot x (2 * (k / 2) + k % 2) (Masked.innerOff (i - rsched.size))
    rw [Nat.div_add_mod k 2, Masked.repSlot_eq hs hz, repStartZk_eq spec _ hk.le]
    unfold Masked.innerOff
    omega
  · show Masked.repBase x (2 * (k / 2) + k % 2) ≤ q + 1
    rw [Nat.div_add_mod k 2, Masked.repBase_eq hs, repStartZk_eq spec _ hk.le]
    omega
  · show q + 1 < Masked.repBase x (2 * (k / 2) + k % 2 + 1)
    rw [Nat.div_add_mod k 2, Masked.repBase_eq hs, FlockComplete.ZkTables.repStart_succ, hrR,
      repStartZk_eq spec _ hk.le]
    omega
  · intro e he
    show loopNr x (Masked.repSlot x (2 * (k / 2) + k % 2) (Masked.innerOff e)) = _
    have hf4 : [1, _root_.Flock.Firewall.Session.INNER_CONS, 1, _root_.Flock.Zk.Q].filter (· != 0) =
        [1, _root_.Flock.Firewall.Session.INNER_CONS, 1, _root_.Flock.Zk.Q] := by
      decide
    have hNe : FlockComplete.piopRoundsZk spec (Masked.stsOf x)[k / 2]! + Masked.innerOff e y.sch = N + e := by
      rw [hN]
      unfold Masked.innerOff
      omega
    have hie : N + e < (roundsOf spec (Masked.stsOf x)).getD k 0 := by
      rw [hrq]
      omega
    rw [Nat.div_add_mod k 2, Masked.repSlot_eq hs hz, Nat.add_assoc, hNe, loopNr_rep hs hk hie,
      Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, Count.repPlan_list (spec := spec) hz', hf4]
    exact getD_append_at (by rw [← hA]; exact hFO)

end Loop

end Claims.ZKServer
