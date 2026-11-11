module

public import Proofs.Flock.SessionLoop.CutInnerLive
public import Proofs.Flock.SessionLoop.TakesTau
public import Proofs.Flock.Claims.ZK.SimInner
public import Proofs.Flock.Claims.ZK.FrameInner
public import Proofs.Flock.SessionLoop.Inner
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
import Proofs.Flock.Claims.ZK.FinishSwInner
import Proofs.Flock.Claims.ZK.FinishSwMk
import Proofs.Flock.Claims.ZK.FrameRankFault
import Proofs.Flock.SessionLoop.FinCommits
import Proofs.Flock.SessionLoop.FinRounds
import Proofs.Flock.SessionLoop.InnerSlot
import Proofs.Flock.SessionLoop.InnerUnrefused
import Proofs.Flock.SessionLoop.LiveInner
import Proofs.Flock.SessionLoop.RepSrc

/-!
# `LiveInner` at the session loop's own inner rounds

At the inner proof's rounds (`Loop.InnerAt`) `withInner`'s session loop releases the inner proof's round as the frame
schedules it (`LeanRelease.release_innerAt`), and the switched session sends there the cut session's body at the rep's
switched source (`Masked.runCut_innerBody`). The two agree round by round, on the rep's pads (`Loop.RepSrc`) and the
coins the openings open:

* `Masked.runCut_innerBody_live`: `runCut_innerBody` at a `Live` statement, its schedules and commitments returning.
* `Loop.swMsgAt_innerAt`: the switched session's message at rep `2j + r`'s inner round `e < 4`, where the coins below
  it show no refusal, is `innerCutBody` at the rep's switched source, tagged `0`.
* `coinAt_inner`: the coins the frame's own inner round reads (`ε`, `β`) are the opened coins at their slots.
* `ownFrames_one`, `ownFrames_three`: the ZK server's own rounds 1 and 3 at a configuration on given pads.
* `LeanRelease.ofLoop_start_at`: where the frame takes a rep's coins, the rep's frame starts.
* `LeanRelease.liveInner_withInner_at`: `LiveInner`'s equation after one history, where the worker's `final_c'` and `τ`
  commitment are `vs`'s there; `LiveInnerOwn` reads it at the worker's own histories.
* **`LeanRelease.liveInner_withInner`**: `LiveInner` at `withInner`, at a `Live` statement the loop serves, on the rep's
  own tables (`Count.RealTabs`) and pads (`Loop.RepSrc`), where the worker's `final_c'` and `τ` commitment are `vs`'s
  (`Loop.TakesFinalC`, `Loop.TakesTau`) and the rows hold at `vs` (`Loop.RowsOK`). Round 0 is the worker's `final_c'`,
  which is the switched source's masked `final_c`; round 1 the pair's openings at the rep's pads and `ε`; round 2 the
  worker's `τ` commitment, the switched source's; round 3 `Y` at the rep's pads and `β`, the switched source's where the
  rows hold.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

/-- **Rep `2j + r`'s inner round `e` is `e` rounds past its inner round 0.** -/
theorem repSlot_innerOff {x : Msg} {s0 : _} {spec : _root_.Flock.SessionSpec}
    (hsx : FlockDeploy.setupOf true x = some (s0, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2) (e : ℕ) :
    repSlot x (2 * j.val + r.val) (innerOff e) = repSlot x (2 * j.val + r.val) (innerOff 0) + e := by
  have hjs : j.val < (stsOf x).size := lt_of_jOf j
  have hr2 := r.isLt
  have hz₁ : _root_.Flock.Zk.schedule (stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((zkTab x j).sch, (zkTab x j).tPad) := by
    rw [show (2 * j.val + r.val) / 2 = j.val by omega, getElem!_pos (stsOf x) j.val hjs]
    exact (zkTab x j).hz
  rw [repSlot_eq hsx hz₁, repSlot_eq hsx hz₁]
  unfold innerOff
  omega

set_option maxHeartbeats 1000000 in
/-- **`runCut_innerBody` at a `Live` statement**: the tables' schedules return (`schedsPZk_live`), level 0's
commitments and the rep's pads commitment return and send nothing (`commit0Cut_cap`, `padsTreeCut_ok`, quiet), so the
cut session stopped at rep `2j + r`'s inner round `i`, where the coins below it show no refusal, sends there
`innerCutBody` at the rep's source. -/
theorem runCut_innerBody_live {x : Msg} (hx : Live x) (c : Fin (refK x) → FlockDeploy.Coin)
    (src : ℕ → FlockComplete.TabSrc) {s0 : _} {spec : _root_.Flock.SessionSpec}
    (hsx : FlockDeploy.setupOf true x = some (s0, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2) {i : ℕ}
    (hi : i < 4) (hN : ¬ Refused x c (repSlot x (2 * j.val + r.val) (innerOff i))) :
    (runCut x c (repSlot x (2 * j.val + r.val) (innerOff i)) spec src).2[repSlot x (2 * j.val + r.val) (innerOff i)]! =
      FlockComplete.CutFree.innerCutBody (((src j.val).rep r.val).masked (innerLayE x j).fc) ((src j.val).rep r.val)
        _root_.Flock.Firewall.Session.INNER_CONS i := by
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  obtain ⟨schs, hs, -⟩ := schedsPZk_live hx
  have hms : ∀ k < (FlockDeploy.tabsOf (stsOf x)).size,
      (FlockDeploy.tabsOf (stsOf x))[k]!.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by
    intro k hk
    have hk' : k < (stsOf x).size := by simpa [FlockDeploy.tabsOf] using hk
    rw [getElem!_pos (FlockDeploy.tabsOf (stsOf x)) k hk]
    simp only [FlockDeploy.tabsOf, Array.getElem_map]
    exact (hall k hk').2.2
  have hcm : ∀ k, ∃ cm, (FlockComplete.commit0Cut
      (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash : _root_.Flock.Hashes _root_.Flock.V)
      _root_.Flock.MerkleScheme.hm96Sha512 (schs[k]!.1.levels.getD 0 default) (src k) : _root_.Flock.V _) = .ok cm :=
    fun k => (FlockComplete.commit0Cut_cap (schs[k]!.1.levels.getD 0 default) (src k)).imp fun _ h => h.1
  choose cmOf hcmOf using hcm
  have hq : FlockComplete.quietS.R
      ((Array.range (FlockDeploy.tabsOf (stsOf x)).size).mapM fun k =>
        FlockComplete.commit0Cut
          (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash : _root_.Flock.Hashes _root_.Flock.V)
          (FlockDeploy.tabsOf (stsOf x))[k]!.merkle (schs[k]!.1.levels.getD 0 default) (src k))
      ((Array.range (FlockDeploy.tabsOf (stsOf x)).size).mapM fun k =>
        FlockComplete.commit0Cut
          (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash : _root_.Flock.Hashes FlockComplete.PS)
          (FlockDeploy.tabsOf (stsOf x))[k]!.merkle (schs[k]!.1.levels.getD 0 default) (src k)) :=
    _root_.Flock.With.Sim.mapM_array FlockComplete.quietS _ fun k =>
      FlockComplete.CutEq.commit0Cut_quiet _ _ _ _
  have hV : ((Array.range (FlockDeploy.tabsOf (stsOf x)).size).mapM fun k =>
        FlockComplete.commit0Cut
          (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash : _root_.Flock.Hashes _root_.Flock.V)
          (FlockDeploy.tabsOf (stsOf x))[k]!.merkle (schs[k]!.1.levels.getD 0 default) (src k)) =
      .ok ((Array.range (FlockDeploy.tabsOf (stsOf x)).size).map cmOf) :=
    FlockComplete.mapM_ok_mem cmOf _ fun k hk => by
      rw [hms k (by simpa using hk)]
      exact hcmOf k
  obtain ⟨ic, hic⟩ := FlockComplete.padsTreeCut_ok
    (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
      (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0).len + _root_.Flock.Zk.K_PAD).nextPowerOfTwo *
      2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE) ((src j.val).rep r.val)
  exact runCut_innerBody hx c src hsx hs (cms := (Array.range (FlockDeploy.tabsOf (stsOf x)).size).map cmOf)
    (fun l => by rw [hq l, hV]) j r hi hN (ic := ic)
    (fun l => by rw [FlockComplete.CutEq.padsTreeCut_quiet _ _ _ l, hic])

end Masked

end Claims.ZKMal

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)
open _root_.Flock.Transcript (header)
open _root_.Flock.Firewall.Frame (Cfg Masker INNER_ROUNDS labelFrame wordBytes words)
open _root_.Flock.Firewall.Finish (Tab)
open FlockSoundness.Refine (Frame encs)

/-! ### The cut session's inner bodies and the ZK server's own rounds -/

theorem innerCutBody_one (fc : F128) (s : FlockComplete.RepSrc) (N : ℕ) :
    FlockComplete.CutFree.innerCutBody fc s N 1 = encs [.vec [s.rhoIn, s.sigmaIn]] ++ (Frame.sq 2 N).enc :=
  rfl

theorem innerCutBody_three (fc : F128) (s : FlockComplete.RepSrc) (N : ℕ) :
    FlockComplete.CutFree.innerCutBody fc s N 3 = encs [.vec s.y.toList] ++ (Frame.sq 2 _root_.Flock.Zk.Q).enc :=
  rfl

/-- Each inner body is frames, then a squeeze. -/
theorem innerCutBody_sq (fc : F128) (s : FlockComplete.RepSrc) (N : ℕ) :
    ∀ e, ∃ (A : List UInt8) (k : UInt8) (n : ℕ), FlockComplete.CutFree.innerCutBody fc s N e = A ++ (Frame.sq k n).enc
  | 0 => ⟨_, _, _, rfl⟩
  | 1 => ⟨_, _, _, rfl⟩
  | 2 => ⟨_, _, _, rfl⟩
  | _ + 3 => ⟨_, _, _, rfl⟩

theorem innerCutBody_ne_nil (fc : F128) (s : FlockComplete.RepSrc) (N e : ℕ) :
    FlockComplete.CutFree.innerCutBody fc s N e ≠ [] := by
  obtain ⟨A, k, n, h⟩ := innerCutBody_sq fc s N e
  rw [h]
  intro h'
  have hl := congrArg List.length h'
  rw [List.length_append, enc_sq, FlockSoundness.Refine.Frame.header_length, List.length_nil] at hl
  omega

/-- **An inner body starts with no domain frame**, so `dropDom` keeps it. -/
theorem dropDom_innerCutBody (fc : F128) (s : FlockComplete.RepSrc) (N : ℕ) :
    ∀ e, dropDom (0 :: FlockComplete.CutFree.innerCutBody fc s N e) = 0 :: FlockComplete.CutFree.innerCutBody fc s N e
  | 0 => dropDom_noDom [.label _root_.Flock.Zk.LABEL_INNER.toUTF8, .one fc] 1 1 fun f hf => by
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hf
      rcases hf with rfl | rfl <;> trivial
  | 1 => dropDom_noDom [.vec [s.rhoIn, s.sigmaIn]] 2 N fun f hf => by
      rw [List.mem_singleton.mp hf]
      trivial
  | 2 => dropDom_noDom [.bytes s.tauCom] 1 1 fun f hf => by
      rw [List.mem_singleton.mp hf]
      trivial
  | _ + 3 => dropDom_noDom [.vec s.y.toList] 2 _root_.Flock.Zk.Q fun f hf => by
      rw [List.mem_singleton.mp hf]
      trivial

/-- **The ZK server's own round 1 on the pads `P` at the layout `Ly`**: the pair's openings at `ε`. -/
theorem ownFrames_one {c : Cfg} {ε β : F128} {w t : ByteArray} {P : FlockComplete.PadsZk}
    {Ly : _root_.Flock.Zk.PadLayout} (hh : c.h = P.h) (hl : c.lay = Ly) :
    ownFrames c { eps := ε, beta := β } w t 1 = [.vec [ε * P.h[Ly.fa]! + P.h[Ly.tx]!, P.h[Ly.fb]! + P.h[Ly.ty]!]] := by
  subst hl
  rw [← hh]
  rfl

/-- **The ZK server's own round 3**: `Y`, each row's `h` plus `β` times its `μ`. -/
theorem ownFrames_three {c : Cfg} {ε β : F128} {w t : ByteArray} (hsz : c.rowMu.size = c.rowH.size) :
    ownFrames c { eps := ε, beta := β } w t 3 =
      [.vec ((Array.range c.rowH.size).map fun i => c.rowH[i]! + β * c.rowMu[i]!).toList] := by
  have e : ((c.rowH.zip c.rowMu).map fun p => p.1 + β * p.2) =
      (Array.range c.rowH.size).map fun i => c.rowH[i]! + β * c.rowMu[i]! := by
    apply Array.ext
    · simp [hsz]
    · intro i h1 _
      simp only [Array.size_map, Array.size_zip, hsz, Nat.min_self] at h1
      simp only [Array.getElem_map, Array.getElem_zip, Array.getElem_range]
      rw [getElem!_pos c.rowH i h1, getElem!_pos c.rowMu i (by rw [hsz]; exact h1)]
  exact congrArg (fun y : Array F128 => [Frame.vec y.toList]) e

/-- **The coins the frame's own inner round reads are the opened coins**: at a history of `q + 1` openings whose next
round is a rep's round `i ≤ q`, entry `a` of the rep's coins past `s` (`s + a < i`) is the opened coin at its slot
`q - i + s + a + 1`, where the loop asks for coins there. -/
theorem coinAt_inner (x : Msg) (ops : List (Msg × Msg)) (hl : ops.length ≤ refK x) {q i s a p : ℕ}
    (hq : ops.length = q + 1) (hi : i ≤ q) (hsa : s + a < i) (hp : q - i + (s + a) + 1 = p)
    (hn : 0 < Loop.loopNr x p) :
    ((((Array.range i).map fun b => Loop.wordsAt x (ops.map fun p => CoinLayer.rd p.2) (q - i + b + 1)).extract s
      i).getD a #[]).getD 0 0 =
      Masked.slotAt (x := x) ((CoinLayer.session Loop.loopNr Masked.dataE).pad (K := refK x)
        (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2)) p 0 := by
  subst hp
  have hpo : q - i + (s + a) + 1 < ops.length := by omega
  have e1 : (((Array.range i).map fun b => Loop.wordsAt x (ops.map fun p => CoinLayer.rd p.2) (q - i + b + 1)).extract
      s i).getD a #[] = Loop.wordsAt x (ops.map fun p => CoinLayer.rd p.2) (q - i + (s + a) + 1) := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_extract]
    simp [show a < i - s by omega, show s + a < i by omega]
  have hs := Loop.wordsAt_size x ops (s := ops.length) hpo le_rfl
  rw [List.take_length] at hs
  have h0 : 0 < (Loop.wordsAt x (ops.map fun p => CoinLayer.rd p.2) (q - i + (s + a) + 1)).size := by
    rw [hs]
    exact lt_min hn (by decide)
  have e := Loop.wordsAt_slot Masked.dataE x ops hl (e := 0) hpo le_rfl (by rw [List.take_length]; exact h0)
  rw [List.take_length] at e
  rw [e1, FlockSoundness.Refine.getD_of_lt h0, e]

namespace Loop

/-- **The switched session's message at a rep's inner round**: at a `Live` statement, after `cs` openings standing at
rep `2j + r`'s inner round `e < 4` (`repSlot`), below `refK x` and past the link, where the coins show no refusal
below it, the switched session sends `innerCutBody` at the rep's switched source, tagged `0` (`encStop`, `encExe`,
`runCut_innerBody_live`). -/
theorem swMsgAt_innerAt (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals) {x : Msg} (hx : Masked.Live x)
    (w : Flock.Wit) (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce)
    (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape) (cs : List (CoinLayer.session Loop.loopNr Masked.dataE).Coin)
    {s0 : _} {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x))
    (j : Fin (Masked.jOf x)) (r : Fin 2) {e : ℕ} (he : e < 4)
    (hn : cs.length = Masked.repSlot x (2 * j.val + r.val) (Masked.innerOff e)) (hK : cs.length < refK x)
    (h0 : cs.length ≠ 0)
    (hN : ¬ Masked.Refused x ((CoinLayer.session Loop.loopNr Masked.dataE).pad cs) cs.length) :
    (CoinLayer.session Loop.loopNr Masked.dataE).swMsgAt x ((CoinLayer.session Loop.loopNr Masked.dataE).wt x w)
        (vs x w ν) t cs =
      0 :: FlockComplete.CutFree.innerCutBody
        (((Masked.swSrcE x ((CoinLayer.session Loop.loopNr Masked.dataE).pad cs) w
          (vs x w ν ((CoinLayer.session Loop.loopNr Masked.dataE).pad cs)
            ((CoinLayer.session Loop.loopNr Masked.dataE).seenOf x t)) t j.val).rep r.val).masked
          (Masked.innerLayE x j).fc)
        ((Masked.swSrcE x ((CoinLayer.session Loop.loopNr Masked.dataE).pad cs) w
          (vs x w ν ((CoinLayer.session Loop.loopNr Masked.dataE).pad cs)
            ((CoinLayer.session Loop.loopNr Masked.dataE).seenOf x t)) t j.val).rep r.val)
        _root_.Flock.Firewall.Session.INNER_CONS e := by
  have hb := Masked.runCut_innerBody_live hx ((CoinLayer.session Loop.loopNr Masked.dataE).pad cs)
    (Masked.swSrcE x ((CoinLayer.session Loop.loopNr Masked.dataE).pad cs) w
      (vs x w ν ((CoinLayer.session Loop.loopNr Masked.dataE).pad cs)
        ((CoinLayer.session Loop.loopNr Masked.dataE).seenOf x t)) t) hsx j r he (by rw [← hn]; exact hN)
  unfold RefSession.swMsgAt
  rw [if_pos hK, enc_finS_eq]
  unfold Masked.encStop
  rw [if_neg hN]
  unfold Masked.encExe
  rw [if_neg h0, hsx]
  dsimp only
  rw [hn]
  split
  · exact congrArg (List.cons 0) hb
  · rename_i hlt
    exact absurd (hb.symm.trans (getElem!_neg _ _ hlt)) (innerCutBody_ne_nil _ _ _ _)

end Loop

namespace LeanRelease

open Loop

/-- **Where the frame takes a rep's coins, the rep's frame starts**: where the frame takes the coins after `e0 :: h1`
and the next round is rep stream `k`'s round `j + 1`, the rep's start (`Drive.start` at table `k / 2`'s statement)
returns (`placeholder_ok`). -/
theorem ofLoop_start_at (T : CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (x : Msg) (ρ : (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) {m : Masker}
    (hA : (ofLoop T L₀ real).answered x ρ (e0 :: h1) = .ok m)
    {cs : List FlockDeploy.Coin} (hcs : h1.map (fun p => CoinLayer.rd p.2) = cs)
    {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q k j : ℕ}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h1.length = q + 1)
    (hki : CoinLayer.locate (roundsOf spec sts) 0 q = (k, j + 1)) :
    ∃ lay mask, _root_.Flock.Firewall.Drive.start sts[k / 2]! = .ok (lay, mask) := by
  have hq1 : cs.length = q + 1 := by rw [← hcs, List.length_map, hq]
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
  obtain ⟨lay, mask, hst, -⟩ := placeholder_ok hph'
  exact ⟨lay, mask, hst⟩

set_option maxHeartbeats 2000000 in
/-- **`LiveInner`'s equation at `withInner`, after one history**: at a `Live` statement the loop serves, on the rep's
own tables where `real` answers the loop's (`hR`) and the rep's pads the tape's (`Loop.RepSrc`), after a history
`LiveInner` reads at which the worker's `final_c'` and `τ` commitment are `vs`'s (`hF` and `hT`: `Loop.TakesFinalC`'s and
`Loop.TakesTau`'s conclusions there) and the rows hold at `vs` (`Loop.RowsOK`), the release at the inner proof's round
is the switched session's message. The release is the frame's own round (`release_innerAt`) at rep `2j + r`'s inner
round `e` (`innerAt_slot`), and the switched session's message there `innerCutBody` at the rep's switched source
(`swMsgAt_innerAt`, the coins below it showing no refusal: `innerAt_unrefused`). Round 0: the worker's `final_c'` plus
the pad at `final_c` is `vs`'s value there (`hF`), the source's masked `final_c` less that pad
(`swSrcE_masked_vsWord`). Round 1: the configuration's pads are the tape's (`cfg_tapeOfDraws`, `RepSrc.pads`), at the
rep's layout (`lay_ofLoopLink`), and `ε` the opened coin at the rep's inner round 0 (`coinAt_inner`), so the round is
the pair's openings (`swSrcE_inner`, `innerTripleE`). Round 2: the worker's `τ` commitment is `vs`'s (`hT`), the
switched source's. Round 3: `Y` at the tape's pads and `β`, the opened coin at the rep's `β` round, the switched
source's where the rows hold (`swSrcE_y`, `innerYE`). -/
theorem liveInner_withInner_at (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
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
    (hS : Loop.RepSrc (withInner Masked.dataE L₁ real fin) src x w)
    (wk : (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).toSrv.Worker)
    (hrows : Loop.RowsOK vs x w)
    (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (κ : (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) (v : Msg) (h0 : 0 < ops.length) (hlt : ops.length < refK x)
    (hl : (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true)
    (hok : ((allOk ((CoinLayer.session Loop.loopNr Masked.dataE).refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval
      CoinOpen.shaH).1 = true)
    (hr : (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false)
    (hv : wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false))
    (hI : Loop.InnerAt (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2))
    (hF : ∀ (w' : ByteArray) (j : Fin (Masked.jOf x)) (r : Fin 2), w'.size = 16 →
      v = (_root_.Flock.Firewall.Frame.labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++
        _root_.Flock.Transcript.header 3 1 1 ++ w' ++ _root_.Flock.Transcript.header 4 1 1).data.toList →
      (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).rep x (ops.length + 1) = 2 * j.val + r.val →
      ops.length = Masked.repSlot x (2 * j.val + r.val) (Masked.innerOff 0) →
      v = (Loop.inner0 (wordBytes (words ((fin x w (ν, t) ((m₀, ρ₀) :: ops)).getD 0 .empty)
        ((labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8).size + 16) 1))).data.toList →
      finalCOf ⟨v.toArray⟩ +
          ((ofLoopLink (withInner Masked.dataE L₁ real fin) src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).h.getD
            ((ofLoopLink (withInner Masked.dataE L₁ real fin) src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).lay.fc 0 =
        Loop.vsWord vs x w ν ((CoinLayer.session Loop.loopNr Masked.dataE).pad
          (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2)) t j r
          ((ofLoopLink (withInner Masked.dataE L₁ real fin) src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).lay.fc)
    (hT : ∀ (m : Masker) (w' : ByteArray) (j : Fin (Masked.jOf x)) (r : Fin 2),
      (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).answered x (ν, t) ((m₀, ρ₀) :: ops) = .ok m →
      m.inner = some 2 → w'.size = 64 →
      v = (_root_.Flock.Transcript.header 5 0 64 ++ w' ++ _root_.Flock.Transcript.header 4 1 1).data.toList →
      (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).rep x (ops.length + 1) = 2 * j.val + r.val →
      ops.length = Masked.repSlot x (2 * j.val + r.val) (Masked.innerOff 2) →
      v = (Loop.inner2 (wordBytes (words ((fin x w (ν, t) ((m₀, ρ₀) :: ops)).getD 2 .empty) 16 4))).data.toList →
      w' = Loop.vsTau vs x w ν ((CoinLayer.session Loop.loopNr Masked.dataE).pad
        (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2)) t j r) :
    (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).release x (ν, t) ((m₀, ρ₀) :: ops) false v =
      dropDom ((CoinLayer.session Loop.loopNr Masked.dataE).swMsgAt x
        ((CoinLayer.session Loop.loopNr Masked.dataE).wt x w) (vs x w ν) t
        (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2)) := by
  have hN := innerAt_unrefused L₁ src real fin x w hx hSv hR wk ν t κ m₀ ρ₀ ops v h0 hlt hl hr hv hI
  obtain ⟨rr, hin, hrel, hvr⟩ := release_innerAt L₁ src real fin x w hSv hR wk ν t κ m₀ ρ₀ ops v h0 hlt hl hv hI
  have hlast : (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).last x = refK x + 1 :=
    loop_last loopNr Masked.dataE _ x
  have hkd : (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).kindAt x ((m₀, ρ₀) :: ops).length =
      .masked :=
    kindAt_masked_of _ (by rw [List.length_cons]; omega) (by rw [hlast, List.length_cons]; omega)
  obtain ⟨m, -, -, ha, -⟩ := check_of_live (ofLoopLink (withInner Masked.dataE L₁ real fin) src real) wk x w
    (ν, t) κ ((m₀, ρ₀) :: ops) v hl hkd hv
  have hne : ((m₀, ρ₀) :: ops).length ≠ lastOf x := by rw [lastOf_eq, List.length_cons]; omega
  obtain ⟨s0, spec, sts, q, k, i, tr, hr', rsched, hxs, hq, hloc, hreal, hrs, hle⟩ := hI
  have hq1 : ops.length = q + 1 := by rw [← hq, List.length_map]
  obtain ⟨jf, rf, hk, he4, hslot, -, -, hi0, hiq, hnr⟩ := Loop.innerAt_slot hx hSv hxs hq
    (by rw [List.length_map]; exact hlt) hloc (hR (ν, t) s0 spec sts hxs) hreal hrs hle
  have hr2 := rf.isLt
  have hsts : Masked.stsOf x = sts := by
    unfold Masked.stsOf
    rw [hxs]
  subst hsts
  obtain ⟨j', rfl⟩ : ∃ j', i = j' + 1 := ⟨i - 1, by omega⟩
  have hin' := Loop.innerOf_eq (real x w (ν, t)) ((withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops))
    (fin x w (ν, t) ((m₀, ρ₀) :: ops)) x _ hxs hq hloc hreal hrs hle he4
  obtain rfl := Option.some.inj (hin.symm.trans hin')
  obtain ⟨W, T, hW, hT', hsch⟩ : ∃ W T : ByteArray, W.size = 16 ∧ T.size = 64 ∧
      ∀ (c : Cfg) (s : Masker) (n k : ℕ),
        _root_.Flock.Firewall.Frame.innerSched c s (fin x w (ν, t) ((m₀, ρ₀) :: ops)) n k = ownRound c s W T n k :=
    ⟨wordBytes (words ((fin x w (ν, t) ((m₀, ρ₀) :: ops)).getD 0 .empty)
        ((labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8).size + 16) 1),
      wordBytes (words ((fin x w (ν, t) ((m₀, ρ₀) :: ops)).getD 2 .empty) 16 4),
      by simp only [wordBytes_size, Loop.words_size], by simp only [wordBytes_size, Loop.words_size],
      fun _ _ _ _ => rfl⟩
  have hvr0 := hvr
  rw [hsch] at hrel hvr
  have hcfg : (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).cfg x (ν, t) ((m₀, ρ₀) :: ops) =
      (withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops) := by
    unfold cfg drawsAt
    rfl
  have hrep : (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).rep x (ops.length + 1) =
      2 * jf.val + rf.val := by
    show Loop.repOf x (ops.length + 1) = _
    rw [show ops.length + 1 = q + 2 by omega, repOf_add_two hxs, hloc]
    exact hk
  obtain ⟨lay, mask, hst⟩ := ofLoop_start_at Masked.dataE
    (withFiles (withLink (withInner Masked.dataE L₁ real fin) src) src) real x (ν, t) (m₀, ρ₀) ops ha rfl hxs hq1 hloc
  rw [show k / 2 = jf.val by omega] at hst
  have hlay := lay_ofLoopLink (L₀ := withInner Masked.dataE L₁ real fin) src real jf rf hst
  have hd := draws_ofLoopLink (L₀ := withInner Masked.dataE L₁ real fin) src real
  have hz := hS.size (ofLoopLink (withInner Masked.dataE L₁ real fin) src real) hd ν t jf rf hlay
  have hpads := hS.pads (ofLoopLink (withInner Masked.dataE L₁ real fin) src real) hd ν t jf rf hlay
    (Masked.UE.tapeZk t)
  obtain ⟨hH, hRH, hRM⟩ := cfg_tapeOfDraws (ofLoopLink (withInner Masked.dataE L₁ real fin) src real) x (ν, t)
    ((m₀, ρ₀) :: ops) (Masked.UE.tapeZk t) Loop.frepOf jf.val rf.val hrep hz
  rw [hlay, ← hpads, hcfg] at hH hRH hRM
  have hLay : ((withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).lay = Masked.layE x jf := by
    rw [← hcfg]
    show (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).lay x
      ((ofLoopLink (withInner Masked.dataE L₁ real fin) src real).rep x (ops.length + 1)) = _
    rw [hrep]
    exact hlay
  have hP : FlockComplete.padsOf (Masked.UE.tapeZk t) jf.val rf.val (Masked.layE x jf) =
      Masked.innerPadsE x t jf rf := by
    show _ = FlockComplete.padsOf (Masked.UE.tapeZk t) jf.val rf.val (Masked.innerLayE x jf)
    rw [Masked.innerLayE_layE]
  have hH' : ((withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).h = (Masked.innerPadsE x t jf rf).h := by
    rw [hH, hP]
  have hl' : ((withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).lay = Masked.innerLayE x jf :=
    hLay.trans (Masked.innerLayE_layE x jf).symm
  rw [hrel, Loop.swMsgAt_innerAt vs hx w ν t (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2)
    hxs jf rf he4 (by rw [List.length_map, hq1]; exact hslot) (by rw [List.length_map]; exact hlt)
    (by rw [List.length_map]; omega) (by rw [List.length_map]; exact hN), dropDom_innerCutBody]
  refine congrArg (List.cons 0) ?_
  rw [ownRound_encs hW hT' he4]
  generalize hE : j' + 1 - rsched.size = e at hslot he4 hvr hvr0 ⊢
  obtain rfl | rfl | rfl | rfl : e = 0 ∨ e = 1 ∨ e = 2 ∨ e = 3 := by omega
  · -- round 0: `final_c'`
    have hTF := hF W jf rf hW hvr hrep (hq1.trans hslot) (hvr0.trans rfl)
    rw [hcfg, hLay, hH] at hTF
    have hWb : (readWord W.data.toList).toBytes = W :=
      ByteArray.ext (Array.ext' (toBytes_readWord _ (by rw [Array.length_toList]; exact hW)))
    have e0 : v = (Loop.inner0 (readWord W.data.toList).toBytes).data.toList := by
      rw [hWb]
      exact hvr
    rw [e0, Loop.finalCOf_inner0] at hTF
    have hfcE : (Masked.layE x jf).fc < (Masked.tabC Masked.PE x jf).z.nE := by
      rw [Masked.nE_eq_hab]
      exact Masked.lt_hab _ le_rfl
    have hfcL : (Masked.layE x jf).fc <
        (FlockComplete.padsOf (Masked.UE.tapeZk t) jf.val rf.val (Masked.layE x jf)).h.size := by
      rw [FlockComplete.padsOf_size]
      have := Masked.lt_hab (Masked.layE x jf) (le_refl _)
      show _ < (Masked.layE x jf).hab + 3 + 1
      omega
    rw [Masked.innerLayE_layE, Loop.swSrcE_masked_vsWord vs x w ν
        ((CoinLayer.session Loop.loopNr Masked.dataE).pad
          (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2)) t jf rf hfcE, ← hTF,
      FlockSoundness.Refine.getD_of_lt hfcL, FlockSoundness.Discharge.Composed.ZerocheckMasked.add_add_self]
    rfl
  · -- round 1: the pair's openings
    have hn1 : loopNr x (q + 1) = _root_.Flock.Firewall.Session.INNER_CONS := by
      rw [hslot]
      exact hnr 1 (by decide)
    have hL1 := Masked.repSlot_innerOff hxs jf rf 1
    rw [hn1, ownFrames_one hH' hl',
      coinAt_inner x ops hlt.le hq1 hiq (s := rsched.size) (a := 0)
        (p := Masked.repSlot x (2 * jf.val + rf.val) (Masked.innerOff 0)) (by omega) (by omega)
        (by rw [hnr 0 (by decide)]; decide),
      innerCutBody_one,
      (Masked.swSrcE_inner x _ w (vs x w ν _ ((CoinLayer.session Loop.loopNr Masked.dataE).seenOf x t)) t jf rf).1,
      (Masked.swSrcE_inner x _ w (vs x w ν _ ((CoinLayer.session Loop.loopNr Masked.dataE).seenOf x t)) t jf rf).2,
      (Masked.innerTripleE x ((CoinLayer.session Loop.loopNr Masked.dataE).pad
        (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2)) w t jf rf rfl).1,
      (Masked.innerTripleE x ((CoinLayer.session Loop.loopNr Masked.dataE).pad
        (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2)) w t jf rf rfl).2]
    rfl
  · -- round 2: `τ`'s commitment
    obtain ⟨-, hin2, -⟩ := ofLoop_answered_inner Masked.dataE
      (withFiles (withLink (withInner Masked.dataE L₁ real fin) src) src) real x w hSv hR (ν, t) (m₀, ρ₀) ops ha
      hne rfl hxs hq1 hloc hreal hrs (by omega)
    rw [show j' - rsched.size + 1 = 2 by omega] at hin2
    have hTT := hT m T jf rf ha hin2 hT' hvr hrep (hq1.trans hslot) (hvr0.trans rfl)
    rw [hTT]
    rfl
  · -- round 3: `Y`
    have hnQ : loopNr x (q + 1) = _root_.Flock.Zk.Q := by
      rw [hslot]
      exact hnr 3 (by decide)
    have hL2 := Masked.repSlot_innerOff hxs jf rf 2
    have hL3 := Masked.repSlot_innerOff hxs jf rf 3
    have hrowsAt := hrows ν _ t jf (outer_ne_one_ofLoopLink (withInner Masked.dataE L₁ real fin) src real wk hSv
      (ν, t) κ m₀ ρ₀ ops hlt.le hl hr jf)
    have hY := Masked.cutE_Y1_ne hx hN hxs w t jf rf (by rw [Masked.betaSlot_inner hx hxs]; omega)
    have hRH' : ((withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).rowH =
        (Masked.innerPadsE x t jf rf).h ++ (Masked.innerPadsE x t jf rf).padH := by
      rw [hRH, hP]
    have hRM' : ((withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).rowMu =
        (Masked.innerPadsE x t jf rf).mu ++ (Masked.innerPadsE x t jf rf).padMu := by
      rw [hRM, hP]
    have hsz : ((withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).rowMu.size =
        ((withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).rowH.size := by
      rw [hRH, hRM]
      simp [FlockComplete.padsOf]
    rw [hnQ, ownFrames_three hsz, hRH', hRM',
      coinAt_inner x ops hlt.le hq1 hiq (s := rsched.size) (a := 2)
        (p := Masked.repSlot x (2 * jf.val + rf.val) (Masked.innerOff 0) + 2) (by omega) (by omega)
        (by rw [← hL2, hnr 2 (by decide)]; decide),
      innerCutBody_three,
      Masked.swSrcE_y x _ w (vs x w ν _ ((CoinLayer.session Loop.loopNr Masked.dataE).seenOf x t)) t jf rf hrowsAt,
      Masked.innerYE hxs _ w t jf rf rfl hY]
    rfl

/-- **`LiveInner` at `withInner`**: `liveInner_withInner_at` at every history, where the worker's `final_c'` and `τ`
commitment are `vs`'s at all of them (`Loop.TakesFinalC`, `Loop.TakesTau`). -/
theorem liveInner_withInner (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
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
    (hS : Loop.RepSrc (withInner Masked.dataE L₁ real fin) src x w)
    (wk : (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).toSrv.Worker)
    (hF : Loop.TakesFinalC vs (ofLoopLink (withInner Masked.dataE L₁ real fin) src real) wk x w)
    (hT : Loop.TakesTau vs (ofLoopLink (withInner Masked.dataE L₁ real fin) src real) wk x w)
    (hrows : Loop.RowsOK vs x w) :
    LiveInner vs (withInner Masked.dataE L₁ real fin) src real wk x w := by
  intro ν t κ m₀ ρ₀ ops v h0 hlt hl hok hr hv hI
  exact liveInner_withInner_at vs L₁ src real fin x w hx hSv hR hS wk hrows ν t κ m₀ ρ₀ ops v h0 hlt hl hok hr hv hI
    (fun w' j r hw hs hrep _ _ => hF ν t κ m₀ ρ₀ ops v w' j r h0 hlt hl hok hv hw hs hrep)
    (fun m w' j r ha hi hw hs hrep _ _ => hT ν t κ m₀ ρ₀ ops v m w' j r h0 hlt hl hok hv ha hi hw hs hrep)

end LeanRelease

end Claims.ZKServer
