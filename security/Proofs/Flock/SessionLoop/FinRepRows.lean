module

public import Proofs.Flock.Claims.ZK.FinishFwRep
public import Proofs.Flock.Claims.ZK.FinishFwRows
public import Proofs.Flock.Claims.ZK.FrameUnrefused
public import Proofs.Flock.SessionLoop.FinRepOpen
public import Proofs.Flock.SessionLoop.FinRepXs
public import Proofs.Flock.SessionLoop.Tables
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
import Proofs.Flock.Claims.DeployK

/-! `FinRepRows`: every table's rows hold at its base, from four facts the proof consumes, none a premise of a guarantee. -/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open FlockSoundness.Discharge
open FlockSoundness.Refine (fOf gOf)
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Loop

noncomputable abbrev tabAt (x : Msg) (ops : List (Msg × Msg)) (j : Fin (Masked.jOf x)) :=
  (Masked.tabC Masked.PE x j).T (Masked.czExe Masked.PE x (finCoins x ops) j)

noncomputable abbrev randAtT (x : Msg) (ops : List (Msg × Msg)) (t : Masked.UE.Tape) (j : Fin (Masked.jOf x)) :=
  (tabAt x ops j).randOf ((Masked.UE.tape x t).1 j) 0

def histAt (n : ℕ) : Array Msg := Array.replicate n default

theorem go_of_fst {α β : Type} {z : α × β} {a : α} (h : z.1 = a) : z = (a, z.2) := Prod.ext h rfl

variable (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (real : Msg → Flock.Wit → Tp → ℕ → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray))
  (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit)

def RepRun (ν : finS.Nonce) (t : finS.Tape) (κ : (LeanRelease.ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) (s0 : FlockDeploy.Ins) (spec : SessionSpec) (k : ℕ) (tb : _root_.Flock.ZkServer.Finish.Tab)
    (hi : Array ByteArray) (q : _root_.Flock.ZkServer.Finish.RepIn) (f : ByteArray)
    (ic' : _root_.Flock.ZkServer.Finish.InnerCommit) (l : Array Msg) (p : _root_.Flock.Zk.ZkProof)
    (tr : _root_.Flock.Transcript) (l' : Array Msg) : Prop :=
  FinCtx (LeanRelease.ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops ∧
    FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x) ∧
    RepAt L₀.draws src x w (ν, t) ((m₀, ρ₀) :: ops) spec k tb hi q f ∧
    (_root_.Flock.ZkServer.Finish.innerCommitP (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
      (_root_.Flock.ZkServer.Finish.padsOf q.draws) (_root_.Flock.ZkServer.Finish.saltAt q.draws.salts) :
      V _root_.Flock.ZkServer.Finish.InnerCommit) = .ok ic' ∧
    l.size = Masked.repBase x k ∧
    ((_root_.Flock.ZkServer.Finish.repFwC
        (FlockComplete.asksCut (fun k i => Masked.slotAt (finCoins x ops) k i) (refK x))
        (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) tb.st tb.sch tb.tPad (k % 2)
        (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 tb.g tb.cm0.toC
        (_root_.Flock.ZkServer.Finish.claimsOf q.xs) (_root_.Flock.ZkServer.Finish.padsOf q.draws).toC ic'.toC q.vals
        tb.extra tb.lanePad (fun l => _root_.Flock.ZkServer.Finish.saltAt (q.hiSalts.getD (l - 1) ByteArray.empty))).run
        (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8)).go l =
      (.ok (p, tr), l')

def FinRepMsgs : Prop :=
  ∀ ν t κ m₀ ρ₀ ops s0 spec (j : Fin (Masked.jOf x)) (r : Fin 2) tb hi q f ic' l p tr l',
    RepRun L₀ src real wk x w ν t κ m₀ ρ₀ ops s0 spec (2 * j.val + r.val) tb hi q f ic' l p tr l' →
    ∀ i < (Composed.layOf (Masked.zkTab x j).table (Masked.stAt x j)).zp 64,
      (Composed.msgAt ((tabAt x ops j).tab.onVals.vals r (Masked.wtOf Masked.PE x w j, vs x w ν (finCoins x ops) (finS.seenOf x t) j)
        (randAtT x ops t j).1 (randAtT x ops t j).2.1) i).val = q.vals[i]!

def FinRepLanes : Prop :=
  ∀ ν t κ m₀ ρ₀ ops s0 spec (j : Fin (Masked.jOf x)) (r : Fin 2) tb hi q f ic' l p tr l',
    RepRun L₀ src real wk x w ν t κ m₀ ρ₀ ops s0 spec (2 * j.val + r.val) tb hi q f ic' l p tr l' →
    Masked.LaneVals (Masked.zkTab x j).table (Masked.stAt x j)
      (fun i => (tabAt x ops j).tab.onVals.vals r (Masked.wtOf Masked.PE x w j, vs x w ν (finCoins x ops) (finS.seenOf x t) j)
        (randAtT x ops t j).1 (randAtT x ops t j).2.1 i +
        fOf (_root_.Flock.ZkServer.Finish.padsOf q.draws).h[i.val]!)
      p.proof.ligerito p.level0

def FinRepTarget : Prop :=
  ∀ ν t κ m₀ ρ₀ ops s0 spec (j : Fin (Masked.jOf x)) (r : Fin 2) tb hi q f ic' l p tr l',
    RepRun L₀ src real wk x w ν t κ m₀ ρ₀ ops s0 spec (2 * j.val + r.val) tb hi q f ic' l p tr l' →
    ∀ zp zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃,
      ((FlockComplete.zerocheckCut (FlockComplete.asksCut (fun k i => Masked.slotAt (finCoins x ops) k i) (refK x))
          tb.st.m tb.st.digest (FlockComplete.flat tb.cm0.toC.cap) ic'.toC.root
          (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0)
          fun i => q.vals[i]! + (_root_.Flock.ZkServer.Finish.padsOf q.draws).h[i]!).run
          (_root_.Flock.Transcript.init #[]
            (spec.domain spec.tables[(2 * j.val + r.val) / 2]! ((2 * j.val + r.val) % 2)).toUTF8)).go l =
        (.ok ((zp, zc), t₁), l₁) →
      ((FlockComplete.lincheckCut (FlockComplete.asksCut (fun k i => Masked.slotAt (finCoins x ops) k i) (refK x))
          tb.st.fold tb.st.m (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0)
          (fun i => q.vals[i]! + (_root_.Flock.ZkServer.Finish.padsOf q.draws).h[i]!) zc).run t₁).go l₁ =
        (.ok ((lp, lc), t₂), l₂) →
      ((FlockComplete.ringSwitchCut (FlockComplete.asksCut (fun k i => Masked.slotAt (finCoins x ops) k i) (refK x))
          (fun k => _root_.Flock.ZkServer.Finish.switchVec tb.g
            (_root_.Flock.abcClaims tb.st.m tb.st.kLog zc lc ++ _root_.Flock.ZkServer.Finish.claimsOf q.xs)[k]!)
          (_root_.Flock.abcClaims tb.st.m tb.st.kLog zc lc ++ _root_.Flock.ZkServer.Finish.claimsOf q.xs)).run t₂).go l₂ =
        (.ok ((ss, T, sws), t₃), l₃) →
      fOf T = Composed.openT (Masked.zkTab x j).table (Masked.stAt x j) (Masked.czExe Masked.PE x (finCoins x ops) j) r
        ((tabAt x ops j).tab.onVals.nu (Masked.wtOf Masked.PE x w j, vs x w ν (finCoins x ops) (finS.seenOf x t) j) +
          (tabAt x ops j).tab.onVals.maskHom (randAtT x ops t j).1)

def FinRepRestart : Prop :=
  ∀ ν t κ m₀ ρ₀ ops s0 spec (j : Fin (Masked.jOf x)) (r : Fin 2) tb hi q f ic' l p tr l',
    RepRun L₀ src real wk x w ν t κ m₀ ρ₀ ops s0 spec (2 * j.val + r.val) tb hi q f ic' l p tr l' →
    gOf p.level0.tPrime = ∑ col, (tabAt x ops j).tab.onVals.y1'
      (Masked.wtOf Masked.PE x w j, vs x w ν (finCoins x ops) (finS.seenOf x t) j) (randAtT x ops t j).1 (randAtT x ops t j).2.1.2.1 r col *
        Composed.bFold (Masked.zkTab x j).table (Masked.stAt x j) (Masked.czExe Masked.PE x (finCoins x ops) j) r col.val

theorem proofsOf_ok_of_ctx {ν : finS.Nonce} {t : finS.Tape} {κ m₀ ρ₀ ops}
    (hctx : FinCtx (LeanRelease.ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops) {s0 : FlockDeploy.Ins}
    {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x)) :
    ∃ files, proofsOf src L₀.draws x w (ν, t) ((m₀, ρ₀) :: ops) spec (Masked.stsOf x) = .ok files := by
  have hs := hctx.2.2.2.2.1
  have hs' : (filesE src L₀.draws x w (ν, t) ((m₀, ρ₀) :: ops)).isOk = true := by
    change (!(filesE src L₀.draws x w (ν, t) ((m₀, ρ₀) :: ops)).isOk) = false at hs
    simpa using hs
  cases hfe : filesE src L₀.draws x w (ν, t) ((m₀, ρ₀) :: ops) with
  | error e => rw [hfe] at hs'; cases hs'
  | ok fp =>
  unfold filesE at hfe
  rw [hsx] at hfe
  dsimp only at hfe
  cases hp : proofsOf src L₀.draws x w (ν, t) ((m₀, ρ₀) :: ops) spec (Masked.stsOf x) with
  | error e => rw [hp] at hfe; cases hfe
  | ok files => exact ⟨files, rfl⟩

theorem finRepRows (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Loop.Serves x)
    (hS : Loop.Level0Src src x w) (hM : FinRepMsgs vs L₀ src real wk x w) (hV : FinRepLanes vs L₀ src real wk x w)
    (hT : FinRepTarget vs L₀ src real wk x w) (hTP : FinRepRestart vs L₀ src real wk x w) :
    FinRepRows vs L₀ src real wk x w := by
  have bok : ∀ {α β : Type} {y : V α} {f : α → V β} {b : β}, y >>= f = .ok b → ∃ a, y = .ok a ∧ f a = .ok b := by
    intro α β y f b hb
    cases y with
    | error e => cases hb
    | ok a => exact ⟨a, rfl, hb⟩
  have hx : Masked.Live x := Masked.live_of hsc hR
  intro ν t κ m₀ ρ₀ ops hctx j
  have hj : j.val < (Masked.stsOf x).size := Masked.lt_of_jOf j
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨-, -, hTs, -⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
  have hnr : ¬ Masked.Refused x (finCoins x ops) (refK x) :=
    LeanRelease.frameUnrefused_ofLoop Masked.dataE (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) real
      CoinOpen.shaH wk x w hServe hx ν t κ m₀ ρ₀ ops hctx.1 hctx.2.1 hctx.2.2.1 hctx.2.2.2.1
  obtain ⟨files, hp⟩ := proofsOf_ok_of_ctx L₀ src real wk x w hctx hsx
  obtain ⟨schs, hschs, hschZ⟩ := Masked.schedsPZk_live hx
  refine Composed.rowsAt_of_fw Masked.PE (Masked.zkTab x j).table (Masked.stAt x j)
    (Masked.czExe Masked.PE x (finCoins x ops) j) (Masked.wtOf Masked.PE x w j, vs x w ν (finCoins x ops) (finS.seenOf x t) j)
    (randAtT x ops t j).1 (randAtT x ops t j).2.1 fun r => ?_
  have hk : 2 * j.val + r.val < 2 * spec.tables.size := by
    have := r.isLt
    omega
  have hk2 : (2 * j.val + r.val) / 2 = j.val := by
    have := r.isLt
    omega
  have hkr : (2 * j.val + r.val) % 2 = r.val := by
    have := r.isLt
    omega
  have hrs : spec.repStreams.size = 2 * spec.tables.size := FlockDeploy.repStreams_size spec
  obtain ⟨tb, hi, q, hRep⟩ := proofsOf_at hp (k := 2 * j.val + r.val) (by rw [hrs]; exact hk)
  obtain ⟨ic', p, tr, hic', hrun, -⟩ := _root_.Flock.ZkServer.Finish.proofFile_ok hRep.2.2
  have hl : (histAt (Masked.repBase x (2 * j.val + r.val))).size = Masked.repBase x (2 * j.val + r.val) :=
    Array.size_replicate
  have hgo := go_of_fst (finRepCut L₀ src real wk x w hsc ν t κ m₀ ρ₀ ops hctx s0 spec hsx _ hk tb hi q _ hRep ic' p tr
    hic' hrun _ hl)
  have hrr : RepRun L₀ src real wk x w ν t κ m₀ ρ₀ ops s0 spec (2 * j.val + r.val) tb hi q _ ic' _ p tr _ :=
    ⟨hctx, hsx, hRep, hic', hl, hgo⟩
  have hM' := hM ν t κ m₀ ρ₀ ops s0 spec j r tb hi q _ ic' _ p tr _ hrr
  have hV' := hV ν t κ m₀ ρ₀ ops s0 spec j r tb hi q _ ic' _ p tr _ hrr
  have hT' := hT ν t κ m₀ ρ₀ ops s0 spec j r tb hi q _ ic' _ p tr _ hrr
  have hTP' := hTP ν t κ m₀ ρ₀ ops s0 spec j r tb hi q _ ic' _ p tr _ hrr
  obtain ⟨htb, hsb⟩ := finRepTab (L₀ := L₀) (src := src) (x := x) (w := w) hsc hsx hschs hk hRep
  rw [hk2] at htb hsb
  have hst : tb.st = _root_.Flock.Setup.ofCircuit (Masked.stAt x j) := htb.symm.trans (Masked.tabsOf_stAt x j)
  have hsch : tb.sch = (Masked.zkTab x j).sch := (Prod.mk.inj (hsb.symm.trans (hschZ j))).1
  obtain ⟨-, -, -, hg⟩ := hS.realOf ν t j r hRep.1
  obtain ⟨-, hps, hab, htw⟩ := finRepPads L₀ src real wk x w hsc ν t κ m₀ ρ₀ ops hctx s0 spec hsx _ hk tb hi q _ hRep
  have hlay : _root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0 =
      Composed.layOf (Masked.zkTab x j).table (Masked.stAt x j) := by
    rw [hst, hsch]
    rfl
  rw [hlay] at hps hab htw
  have hxsz : (_root_.Flock.ZkServer.Finish.claimsOf q.xs).size = spec.points * (Masked.stAt x j).regions.size := by
    have hq := hRep.2.1
    unfold repInOf at hq
    obtain ⟨⟨lay, ms⟩, -, hq⟩ := bok hq
    have hqx : q.xs = _root_.Flock.ZkServer.Drive.extras (Masked.stsOf x)[(2 * j.val + r.val) / 2]!
        (_root_.Flock.ZkServer.Drive.linkPoints (Masked.stsOf x)[(2 * j.val + r.val) / 2]!
          (wordsAt x (ops.map fun p => CoinLayer.rd p.2) 0)) := by
      cases hq
      rfl
    rw [linkPoints_wordsAt x w hsc hR hctx.1 hsx hk m₀ ρ₀, hk2,
      show (Masked.stsOf x)[j.val]! = Masked.stAt x j from getElem!_pos (Masked.stsOf x) j.val hj] at hqx
    simp only [_root_.Flock.ZkServer.Finish.claimsOf, Array.size_map]
    rw [hqx]
    unfold _root_.Flock.ZkServer.Drive.extras
    rw [flatMap_size_const _ _ _ fun _ => Array.size_map]
    simp only [viewPoints, Array.size_map, Array.size_range]
    exact Nat.mul_comm _ _
  rw [hkr, hk2, hsch] at hgo
  rw [hkr, hk2, hsch] at hT'
  have ht₀ : (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[j.val]! r.val).toUTF8).rounds = #[] := rfl
  exact Masked.fwRows_at x hx w hR t (finCoins x ops) (refK x) hnr s0 spec hsx j r
    (vs x w ν (finCoins x ops) (finS.seenOf x t)) (_root_.Flock.ZkServer.Finish.claimsOf q.xs) hxsz tb.st hst tb.tPad _ tb.g hg tb.cm0.toC
    (_root_.Flock.ZkServer.Finish.padsOf q.draws).toC ic'.toC q.vals tb.extra tb.lanePad _ hps hab htw ht₀ hl hgo
    hM' hV' hT' hTP'

end Loop

end Claims.ZKServer
