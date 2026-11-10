module

public import Proofs.Flock.SessionLoop.Inner
public import Proofs.Flock.SessionLoop.TakenVals
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
import Proofs.Flock.Claims.Complete.ZkTables.Table
import Proofs.Flock.Claims.ZK.FrameBeta
import Proofs.Flock.Claims.ZK.FrameOuter
import Proofs.Flock.Claims.ZK.FrameRankFault
import Proofs.Flock.Claims.ZK.FrameRho
import Proofs.Flock.Claims.ZK.FrameTaken
import Proofs.Flock.SessionLoop.CutPiopLive
import Proofs.Flock.SessionLoop.InnerSlot
import Proofs.Flock.SessionLoop.LiveInner

/-!
# The loop's frames refuse every coin fault shown below the inner proof's rounds

At `Loop.InnerAt` the history stands at a rep's inner round, within the rep's rounds and past its whole schedule
(`Loop.innerAt_slot`), so every fault the coins below it show is one a frame refused:

* `Loop.check_inner0_inner`: the frame's check of the inner proof's round 0, passed or stood in, leaves the frame in the
  inner proof.
* **`LeanRelease.unrefused_rep`**: on a live history whose last coins the frame takes, no refusal shows by a round the
  history holds within rep `2j + r`'s rounds, where the rep's rank check and `ρ` pair hold at the opened coins. An
  outer coin shown there is answered within the history (`outer_unrefused`); an earlier rep's rounds all are, and the
  frames refused its faults (`rank_unrefused`, `rho_unrefused`, `beta_unrefused`); the rep's own `β` round, where shown,
  is within the history; and no later rep's round is below its first (`slots_order`).
* **`LeanRelease.innerAt_unrefused`**: at `LiveInner`'s premises at `withInner`, no refusal shows by the history's
  length. The rep's rank check and `ρ` pair hold where the loop's state at the rep's inner round 0 is inside the inner
  proof or owes coins (`rank_of_prog`, `rho_of_prog`): past round 0 the frame took that round's coins
  (`coins_taken_of_live`); at round 0 the frame's check put the frame in the inner proof (`check_inner0_inner`), and the
  loop's state one round on is the check's (`ofLoop_progChecks_sizes`, at `withInner`'s round 0, the ZK server's own:
  `LoopInnerRound0Own`).
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)
open _root_.Flock.Transcript (header)
open _root_.Flock.Firewall.Frame (Cfg Masker INNER_ROUNDS check labelFrame wordBytes words)
open _root_.Flock.Firewall.Finish (Tab)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

/-- **The inner proof's round 0 puts the frame in the inner proof**: where the frame's check of a device round against
the ZK server's own round 0 returns, whether the device's round passed or the round stood in, the frame is at the inner
proof's round 0. -/
theorem check_inner0_inner {c : Cfg} {m s' : Masker} {w' : ByteArray} (hw : w'.size = 16) {n : ℕ}
    {v r out : ByteArray} {why : Option String}
    (hr : r.data.toList =
      (labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 ++ w' ++ header 4 1 1).data.toList)
    (h : check c m v r n = .ok (out, s', why)) : s'.inner = some 0 := by
  have ht : (ByteArray.mk (Array.replicate 64 0)).size = 64 := by simp [ByteArray.size]
  have e : _root_.Flock.Firewall.Frame.ownRound c {} w' (ByteArray.mk (Array.replicate 64 0)) n 0 =
      labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 ++ w' ++ header 4 1 1 := rfl
  have hR : r = labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 ++ w' ++ header 4 1 1 := by
    have e2 : r = ByteArray.mk r.data.toList.toArray := by rw [Array.toArray_toList]
    rw [e2, hr, Array.toArray_toList]
  subst hR
  cases why with
  | none =>
    obtain ⟨fr, hf⟩ := Sched.roundFree_round (check_none h)
    rw [(check_inner0 hw h).1, ← e] at hf
    exact roundFree_own0_inner hw ht hf
  | some _ =>
    obtain ⟨fr, hf⟩ := Sched.roundFree_round (check_some h)
    rw [← e] at hf
    exact roundFree_own0_inner hw ht hf

end Loop

namespace LeanRelease

open Loop

/-- **The loop's frames refuse every fault shown within a rep's rounds where its rank check and `ρ` pair hold**: on a
live history of at most `refK x` openings whose last coins the frame takes, no refusal shows by a round `N` the history
holds from rep `2j + r`'s first round to below the next rep's, where the rep's rank check and `ρ` pair hold at the
opened coins. An outer coin shown there is answered within the history (`outer_unrefused`); an earlier rep's rounds
are all below `N` (`slots_order`), so the frames refused its rank check, `ρ` pair and `β` (`rank_unrefused`,
`rho_unrefused`, `beta_unrefused`); the rep's own `β`, where shown, is answered within the history
(`beta_unrefused`); and a later rep's rank check, `ρ` pair and `β` are past its first round. -/
theorem unrefused_rep (T : ZKMal.CoinLayer.TableData)
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
    {N : ℕ} (hN : N ≤ ops.length) {j r : ℕ} (hj : j < (Masked.stsOf x).size) (hr2 : r < 2)
    (hq0 : Masked.repBase x (2 * j + r) ≤ N) (hq1 : N < Masked.repBase x (2 * j + r + 1))
    (hrank : ¬ Masked.RankBad (x := x) ((ZKMal.CoinLayer.session loopNr T).pad (K := refK x)
      (ops.map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2)) j r)
    (hrho : _root_.Flock.Zk.rhoIndependent (Masked.rhoAt (x := x) ((ZKMal.CoinLayer.session loopNr T).pad
      (K := refK x) (ops.map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2)) j r) = true) :
    ¬ Masked.Refused x ((ZKMal.CoinLayer.session loopNr T).pad (K := refK x)
      (ops.map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2)) N := by
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (show 0 < (Masked.stsOf x).size by omega)
  have hbase : ∀ {a b : ℕ}, a ≤ b → Masked.repBase x a ≤ Masked.repBase x b := by
    intro a b hab
    rw [Masked.repBase_eq hsx, Masked.repBase_eq hsx]
    exact FlockComplete.ZkTables.repStart_mono spec (Masked.stsOf x) hab
  rintro (⟨j', hj', r', hr', ho, e, he, h1⟩ | ⟨j', hj', r', hr', hth⟩)
  · exact outer_unrefused T L₀ real wk x w hx hSv ρ κ m₀ ρ₀ ops hl hlive hr hj' hr' (by omega) he h1
  · have Sp := Masked.slots_order hx hsx hj' hr'
    rcases Nat.lt_trichotomy (2 * j' + r') (2 * j + r) with hlt | heq | hgt
    · have hb : Masked.betaSlot x j' r' < ops.length := by
        have := hbase (show 2 * j' + r' + 1 ≤ 2 * j + r by omega)
        omega
      rcases hth with ⟨-, -, -, -, hbad⟩ | ⟨-, hd⟩ | ⟨-, hz⟩
      · exact rank_unrefused T L₀ real wk x w hx hSv ρ κ m₀ ρ₀ ops hl hlive hr hj' hr' hb hbad
      · rw [rho_unrefused T L₀ real wk x w hx hSv ρ κ m₀ ρ₀ ops hl hlive hr hj' hr' hb] at hd
        exact Bool.noConfusion hd
      · rw [beta_unrefused T L₀ real wk x w hx hSv ρ κ m₀ ρ₀ ops hl hlive hr hj' hr' hb] at hz
        exact Bool.noConfusion hz
    · obtain ⟨rfl, rfl⟩ : j' = j ∧ r' = r := ⟨by omega, by omega⟩
      rcases hth with ⟨-, -, -, -, hbad⟩ | ⟨-, hd⟩ | ⟨hβ, hz⟩
      · exact hrank hbad
      · rw [hrho] at hd
        exact Bool.noConfusion hd
      · rw [beta_unrefused T L₀ real wk x w hx hSv ρ κ m₀ ρ₀ ops hl hlive hr hj' hr' (by omega)] at hz
        exact Bool.noConfusion hz
    · have := hbase (show 2 * j + r + 1 ≤ 2 * j' + r' by omega)
      rcases hth with ⟨-, -, -, hrs, -⟩ | ⟨hρ, -⟩ | ⟨hβ, -⟩ <;> omega

set_option maxHeartbeats 400000 in
/-- **At the inner proof's rounds the loop's frames refuse every fault the coins show**: at `LiveInner`'s premises at
`withInner`, on the rep's own tables where `real` answers the loop's (`hR`) at a `Live` statement the loop serves, no
refusal shows by the history's length. The history stands at rep `2j + r`'s inner round, within the rep's rounds
(`innerAt_slot`), so it is `unrefused_rep` where the rep's rank check and `ρ` pair hold. They do where the loop's state at
the rep's inner round 0 is inside the inner proof or owes coins (`rank_of_prog`, `rho_of_prog`): past round 0 the frame
took that round's coins (`coins_taken_of_live`); at round 0, the history's next round, the frame's check of the round in
the device round's place, `withInner`'s round 0 (`scheduled_innerAt`), leaves the frame in the inner proof
(`check_inner0_inner`), and the loop's state one round on is that check's (`ofLoop_progChecks_sizes`, its round 0 the
ZK server's own: `LoopInnerRound0Own`). -/
theorem innerAt_unrefused
    (L₁ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (fin : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      List (Msg × Msg) → Array ByteArray)
    (x : Msg) (w : Flock.Wit) (hx : Masked.Live x) (hSv : Serves x)
    (hR : ∀ (ρ : (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
        (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand) (s0 : _) (spec : SessionSpec) (sts : Array Stmt),
      FlockDeploy.setupOf true x = some (s0, spec, sts) → Count.RealTabs sts (real x w ρ))
    (wk : (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).toSrv.Worker)
    (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (κ : (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg))
    (v : Msg) (h0 : 0 < ops.length) (hlt : ops.length < refK x)
    (hl : (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) =
      true)
    (hr : (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false)
    (hv : wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false))
    (hI : Loop.InnerAt (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2)) :
    ¬ Masked.Refused x ((CoinLayer.session Loop.loopNr Masked.dataE).pad (K := refK x)
      (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2)) ops.length := by
  have hsch := schedOf_none_of_innerAt hI
  obtain ⟨s0, spec, sts, q, k, i, tr, hr', rsched, hxs, hq, hloc, hreal, hrs, hle⟩ := hI
  have hq1 : ops.length = q + 1 := by rw [← hq, List.length_map]
  obtain ⟨jf, rf, -, -, hslot, hb0, hb1, -, -, -⟩ := Loop.innerAt_slot hx hSv hxs hq (by rw [List.length_map]; exact hlt)
    hloc (hR (ν, t) s0 spec sts hxs) hreal hrs hle
  have hj : jf.val < (Masked.stsOf x).size := Masked.lt_of_jOf jf
  have hr2 : rf.val < 2 := rf.isLt
  obtain ⟨s, hsd⟩ : ∃ s, s = Masked.repSlot x (2 * jf.val + rf.val)
      (fun sch => sch.k0 + 7 + 11 * (sch.levels.size - 1) + 0) := ⟨_, rfl⟩
  have hs0 : 1 ≤ s ∧ s + (i - rsched.size) = ops.length := by
    obtain ⟨s0', spec', hsx⟩ := Masked.setupOf_of_pos (show 0 < (Masked.stsOf x).size by omega)
    obtain ⟨y⟩ := hx.2 jf.val hj
    have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * jf.val + rf.val) / 2]!.m = .ok (y.sch, y.tPad) := by
      rw [show (2 * jf.val + rf.val) / 2 = jf.val by omega, getElem!_pos _ jf.val hj]
      exact y.hz
    rw [hq1, hslot, hsd, Masked.repSlot_eq hsx hz, Masked.repSlot_eq hsx hz]
    unfold Masked.innerOff
    omega
  have hlast : (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).last x = refK x + 1 :=
    loop_last loopNr Masked.dataE _ x
  have hrr : ¬ Masked.RankBad (x := x) ((CoinLayer.session Loop.loopNr Masked.dataE).pad (K := refK x)
        (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2)) jf.val rf.val ∧
      _root_.Flock.Zk.rhoIndependent (Masked.rhoAt (x := x) ((CoinLayer.session Loop.loopNr Masked.dataE).pad
        (K := refK x) (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2)) jf.val rf.val) =
        true := by
    suffices H : ∃ d, (progOf x ((ops.take s).map (fun p => CoinLayer.rd p.2) ++ [d])).inner.isSome = true ∨
        (progOf x ((ops.take s).map (fun p => CoinLayer.rd p.2) ++ [d])).owed ≠ none by
      obtain ⟨d, ho⟩ := H
      rw [hsd] at ho
      exact ⟨rank_of_prog Masked.dataE x hx hSv ops hlt.le hj hr2 (e := 0) (by decide) (by omega) d ho,
        rho_of_prog Masked.dataE x hx hSv ops hlt.le hj hr2 (e := 0) (by decide) (by omega) d ho⟩
    rcases Nat.eq_zero_or_pos (i - rsched.size) with h0' | hpos
    · have hS : s = ops.length := by omega
      have hkd : (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).kindAt x ((m₀, ρ₀) :: ops).length =
          .masked :=
        kindAt_masked_of _ (by rw [List.length_cons]; omega) (by rw [hlast, List.length_cons]; omega)
      have hin := innerOf_eq (real x w (ν, t)) ((withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops))
        (fin x w (ν, t) ((m₀, ρ₀) :: ops)) x _ hxs hq hloc hreal hrs hle (by rw [h0']; decide)
      rw [h0'] at hin
      obtain ⟨m, out, s', ha, hc⟩ := check_of_live (ofLoopLink (withInner Masked.dataE L₁ real fin) src real) wk x w
        (ν, t) κ ((m₀, ρ₀) :: ops) v hl hkd hv
      have hP : ProgChecks (ofLoopLink (withInner Masked.dataE L₁ real fin) src real) x w :=
        ofLoop_progChecks_sizes Masked.dataE (withFiles (withLink (withInner Masked.dataE L₁ real fin) src) src) real x w
          (sizesOk_of_serves hSv) hR (_root_.Flock.SecurityProofs.LoopInnerRound0Own Masked.dataE L₁ real fin x w)
      obtain ⟨v', st, s'', why, hc', hp⟩ := hP (ν, t) ((m₀, ρ₀) :: ops) ([] : Msg) m out ha hkd ⟨v, s', none, hc⟩
      unfold checkAt at hc'
      rw [scheduled_innerAt L₁ src real fin x w (ν, t) (m₀, ρ₀) ops hsch hin, Array.toArray_toList] at hc'
      have hw : (wordBytes (words ((fin x w (ν, t) ((m₀, ρ₀) :: ops)).getD 0 .empty)
          ((labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8).size + 16) 1)).size = 16 := by
        simp only [wordBytes_size, Loop.words_size]
      have hin0 := check_inner0_inner hw (by rfl) hc'
      have hI0 : ((ofLoopLink (withInner Masked.dataE L₁ real fin) src real).prog x
          (((m₀, ρ₀) :: ops) ++ [(0 :: out.data.toList, ([] : Msg))])).inner = some 0 := by
        rw [hp]
        exact hin0
      change (progOf x ((ops ++ [(0 :: out.data.toList, ([] : Msg))]).map fun p => CoinLayer.rd p.2)).inner =
        some 0 at hI0
      rw [List.map_append, List.map_cons, List.map_nil] at hI0
      refine ⟨CoinLayer.rd ([] : Msg), .inl ?_⟩
      rw [hS, List.take_length]
      exact congrArg Option.isSome hI0
    · obtain ⟨-, m, ha⟩ := coins_taken_of_live (ofLoopLink (withInner Masked.dataE L₁ real fin) src real) wk x w
        (ν, t) κ ((m₀, ρ₀) :: ops) hl (by rw [hlast]; simp only [List.length_cons]; omega) hr (i := s + 2)
        (by omega) (by simp only [List.length_cons]; omega)
      obtain ⟨⟨dm, da⟩, hd⟩ : ∃ d, ops[s]? = some d := ⟨ops[s]'(by omega), List.getElem?_eq_getElem _⟩
      have htk : ops.take (s + 1) = ops.take s ++ [(dm, da)] := by rw [List.take_add_one, hd]; rfl
      have hh : ((m₀, ρ₀) :: ops).take (s + 2) = (m₀, ρ₀) :: (ops.take s ++ [(dm, da)]) := by
        rw [List.take_succ_cons, htk]
      rw [hh] at ha
      change (progOf x ((ops.take s ++ [(dm, da)]).map fun p => CoinLayer.rd p.2)).answer _
        (coinsAt x ((m₀, ρ₀) :: (ops.take s ++ [(dm, da)]))) = .ok m at ha
      rw [List.map_append, List.map_cons, List.map_nil] at ha
      exact ⟨CoinLayer.rd da, .inr fun h => answer_owed_none _ h ha⟩
  exact unrefused_rep Masked.dataE (withFiles (withLink (withInner Masked.dataE L₁ real fin) src) src) real wk x w hx
    hSv (ν, t) κ m₀ ρ₀ ops hlt.le hl hr le_rfl hj hr2 (by omega) (by omega) hrr.1 hrr.2

end LeanRelease

end Claims.ZKServer
