module

public import Proofs.Flock.Claims.ZK.ExePlan
public import Proofs.Flock.Claims.ZK.Stopped
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

/-! The deployed masked session's open premises, as statements. -/

@[expose] public section

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

abbrev hsE : _root_.Flock.Hashes FlockComplete.PS := _root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash

/-- **The deployed session's Ligerito prover of levels ≥ 1.** -/
noncomputable abbrev PE : Composed.LigProver := ofSalts PcExe

/-- **The executable's tape plan** at `PE`. -/
noncomputable abbrev UE : TapePlan (tabC PE) := TapePlan.ofLayout (TapeLayout.exe PE) exeTape

noncomputable abbrev QE : RoundPlan (tabC PE) := RoundPlan.stop (czExe PE) (encExe PE) (finExe PE)

noncomputable def cutE (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape) :
    ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Cut :=
  cutAt (tabC PE x) (czExe PE x c) (wtOf PE x w) (UE.tape x τ).1
    (exeOf (tabC PE x) (czExe PE x c) (wtOf PE x w) (UE.tape x τ).1 (UE.tape x τ).2.1)

theorem cutOf_eq (nr : Msg → ℕ → ℕ) (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (τ : UE.Tape) : (session nr (tableData PE QE UE)).cutOf x c (wtOf PE x w) τ = cutE x c w τ := rfl

noncomputable def srcE (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape) :
    ℕ → FlockComplete.TabSrc :=
  tabSrcCut PE x (czExe PE x c) (cutE x c w τ)

def ioE {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ) : FlockComplete.Asks FlockComplete.PS :=
  FlockComplete.asksCut (fun k i => slotAt c k i) n

def trsE (x : Msg) (w : Claims.Flock.Wit) (ω : FlockComplete.TapeZk) : Array FlockComplete.Trace :=
  FlockComplete.ZkTables.traces (stsOf x) (FlockComplete.ZkTables.maskWOf x w ω.mask)

def runExe (x : Msg) (w : Claims.Flock.Wit) (ω : FlockComplete.TapeZk) (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ)
    (spec : _root_.Flock.SessionSpec) : Except String (String × Array ByteArray) × Array Msg :=
  (FlockComplete.sessionPZk (ioE c n) hsE spec
    (FlockDeploy.tabsOf (stsOf x)) (FlockComplete.MaskRank.masksOf (stsOf x)) (trsE x w ω) ω).go #[]

def liveRunAt (x : Msg) (w : Claims.Flock.Wit) (ω : FlockComplete.TapeZk) (ops : List (Msg × Msg))
    (spec : _root_.Flock.SessionSpec) : Except String (String × Array ByteArray) × Array Msg :=
  (FlockComplete.sessionPZk (FlockComplete.asksS fun k => (ops.map fun p => coinOfD p.2).getD k [])
    hsE spec (FlockDeploy.tabsOf (stsOf x))
    (FlockComplete.MaskRank.masksOf (stsOf x)) (trsE x w ω) ω).go #[]

def ptsE {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (spec : _root_.Flock.SessionSpec) :
    Array (Array _root_.Flock.F128) :=
  (Array.range spec.points).map fun p => (Array.range spec.mPts).map fun i => slotAt c 0 (p * spec.mPts + i)

def priorE {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (j r : ℕ) : Array (List _root_.Flock.F128) :=
  if r = 1 then ptsOpen c (stsOf x)[j]! (2 * j) else #[]

noncomputable def mixE (k : ℕ) (x : Msg) (w : Claims.Flock.Wit) (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin)
    (n : ℕ) (j : Fin (jOf x)) (r : Fin 2) (dom : ByteArray) (cm0 : FlockComplete.Commit)
    (extras : Array _root_.Flock.Claim) (ic : FlockComplete.InnerCommit) :
    _root_.Flock.TMW FlockComplete.PS _root_.Flock.Zk.ZkProof :=
  FlockComplete.repMix k (ioE c n) hsE
    (FlockDeploy.tabsOf (stsOf x))[j.val]! (zkTab x j).sch (zkTab x j).tPad r.val dom
    (trsE x w (UE.tapeZk τ))[j.val]!
    (FlockComplete.packed (trsE x w (UE.tapeZk τ))[j.val]! (FlockDeploy.tabsOf (stsOf x))[j.val]!.m) cm0 extras
    (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val
      (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog
        (zkTab x j).sch.k0)) ic
    ((UE.tapeZk τ).extra j.val) ((UE.tapeZk τ).lanePad j.val) ((UE.tapeZk τ).salt j.val r.val)
    ((UE.tapeZk τ).nonce j.val r.val) (priorE c j.val r.val) (FlockComplete.MaskRank.masksOf (stsOf x))[j.val]!
    ((srcE x c w τ j.val).rep r.val)

/-- **Hybrid `k` sends what hybrid `k + 1` sends** at any opened coins and stop at which no refusal is seen,
but its final-only data. -/
def HybridStep (k : ℕ) : Prop :=
  ∀ x, Live x → ∀ w, RZk x w → ∀ (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ), ¬ Refused x c n →
    ∀ s spec, FlockDeploy.setupOf true x = some (s, spec, stsOf x) →
    ∀ (j : Fin (jOf x)) (r : Fin 2) (dom : ByteArray) (cm0 : FlockComplete.Commit) (ic : FlockComplete.InnerCommit)
      (extras : Array _root_.Flock.Claim),
      (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras →
    ∀ l : Array Msg, l.size = repBase x (2 * j.val + r.val) →
      ((mixE k x w τ c n j r dom cm0 extras ic).run (_root_.Flock.Transcript.init #[] dom)).go l =
        ((mixE (k + 1) x w τ c n j r dom cm0 extras ic).run (_root_.Flock.Transcript.init #[] dom)).go l

def PrefixLive : Prop :=
  ∀ x, Live x → ∀ w, RZk x w → ∀ (ω : FlockComplete.TapeZk) (ops : List (Msg × Msg)), 0 < ops.length →
    ops.length < refK x → ∀ s spec, FlockDeploy.setupOf true x = some (s, spec, stsOf x) →
      (ops.length < (liveRunAt x w ω ops spec).2.size ↔
        ops.length < (runExe x w ω (padOps (x := x) ops) ops.length spec).2.size) ∧
      (ops.length < (liveRunAt x w ω ops spec).2.size →
        (liveRunAt x w ω ops spec).2[ops.length]! =
          (runExe x w ω (padOps (x := x) ops) ops.length spec).2[ops.length]!) ∧
      ((liveRunAt x w ω ops spec).2.size ≤ ops.length → ∃ e, (liveRunAt x w ω ops spec).1 = .error e)

def SimPiop : Prop := HybridStep 0

def SimLig : Prop := HybridStep 1

def SimInner : Prop := HybridStep 2

def SimCommit : Prop :=
  ∀ x, Live x → ∀ w, RZk x w → ∀ (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin) (j : Fin (jOf x)) (l : Array Msg),
    (FlockComplete.commit0Cut hsE
        (FlockDeploy.tabsOf (stsOf x))[j.val]!.merkle ((zkTab x j).sch.levels.getD 0 default) (srcE x c w τ j.val)).go l =
      (FlockComplete.eraseC <$> FlockComplete.commitPZk hsE
        (FlockDeploy.tabsOf (stsOf x))[j.val]!.merkle ((zkTab x j).sch.levels.getD 0 default) (zkTab x j).tPad
        (FlockComplete.packed (trsE x w (UE.tapeZk τ))[j.val]! (FlockDeploy.tabsOf (stsOf x))[j.val]!.m)
        ((UE.tapeZk τ).extra j.val) ((UE.tapeZk τ).lanePad j.val) ((UE.tapeZk τ).salt j.val 0 0)).go l ∧
    ∀ r : Fin 2,
      (FlockComplete.padsTreeCut hsE
          (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
            (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0).len + _root_.Flock.Zk.K_PAD).nextPowerOfTwo *
            2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE) ((srcE x c w τ j.val).rep r.val)).go l =
        (FlockComplete.eraseIC <$> FlockComplete.innerCommitP hsE
          (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val
            (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
              (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0))
          ((UE.tapeZk τ).innerSalt j.val r.val)).go l

/-- **6. The session from the cut sends the session's bodies**, at opened coins and a stop at which no
refusal is seen. -/
def SimSession : Prop :=
  ∀ x, Live x → ∀ w, RZk x w → ∀ (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ), ¬ Refused x c n →
    ∀ s spec, FlockDeploy.setupOf true x = some (s, spec, stsOf x) →
      (runCut x c n spec (srcE x c w τ)).2 = (runExe x w (UE.tapeZk τ) c n spec).2

def CausalExe : Prop := QE.Causal

def FinExe : Prop :=
  ∀ (H : Msg → Msg), Claims.hOf H = _root_.Flock.Sha512.hash → ∀ nr : Msg → ℕ → ℕ,
    (session nr (tableData PE (RoundPlan.stop (czExe PE) (encStop (encExe PE)) (finExe PE)) UE)).HonestFinOn Live RZk H
      coinOfD fun ω => UE.tapeZk ω.2

end Masked

end Claims.ZKMal
