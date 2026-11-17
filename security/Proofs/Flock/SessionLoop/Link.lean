module

public import Proofs.Flock.SessionLoop.Statement
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

/-! The link at the session loop's own level 0 is the recheck's link message, so `LinkRoundEq` holds there. -/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open Claims.ZKMal.CoinLayer Claims.ZKMal.Masked
open _root_.Flock (V F128 Stmt)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

theorem Loop.level0Of_root {R : Type} {src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray)}
    {x : Msg} {w : Flock.Wit} {ρ : R} {sts : Array Stmt} {j : ℕ} {l : _root_.Flock.ZkServer.Level0.Level0}
    (h : Loop.level0Of src x w ρ sts j = .ok l) : l.root = _root_.Flock.ZkServer.Link.root l.toIn := by
  unfold Loop.level0Of at h
  rcases hs : src x w ρ (2 * j) with e | ⟨g, d, hi⟩
  · rw [hs] at h; cases h
  · rw [hs] at h
    exact (_root_.Flock.ZkServer.Level0.ofSamples_ok h).2.2.2.2.2

theorem Loop.level0Of_eq_level0At {R : Type}
    (src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray)) (x : Msg) (w : Flock.Wit) (ρ : R)
    (sts : Array Stmt) (j : ℕ) : Loop.level0Of src x w ρ sts j = Loop.level0At src x w ρ sts (2 * j) := by
  unfold Loop.level0Of Loop.level0At
  rw [Nat.mul_div_cancel_left j (by omega : 0 < 2)]

theorem Loop.Level0Src.linkLevel0 {src : Msg → Flock.Wit →
      (session Loop.loopNr dataE).refP.Nonce × (session Loop.loopNr dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray)} {x : Msg} {w : Flock.Wit}
    (h : Loop.Level0Src src x w) : Loop.LinkLevel0 src x w := by
  intro ν t j
  obtain ⟨l, hl, hin⟩ := h ν t j 0
  exact ⟨l, by rw [Loop.level0Of_eq_level0At]; exact hl, hin⟩

namespace LeanRelease

theorem linkIs_ofLoop
    (L₀ : LeanRelease Flock.Wit ((session Loop.loopNr dataE).refP.Nonce × (session Loop.loopNr dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (session Loop.loopNr dataE).refP.Nonce × (session Loop.loopNr dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (session Loop.loopNr dataE).refP.Nonce × (session Loop.loopNr dataE).refP.Rand →
      ℕ → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray))
    (hL : L₀.link = fun x w ρ _ => Loop.linkOf src x w ρ) {x : Msg} (hx : Live x) {w : Flock.Wit}
    (hs : Loop.LinkLevel0 src x w) : LinkIs Loop.loopNr (ofLoop dataE L₀ real) x w := by
  intro ν t m₀ ρ₀ _ _
  show L₀.link x w (ν, t) [(m₀, ρ₀)] = linkMsg x w t
  rw [hL]
  show Loop.linkOf src x w (ν, t) = linkMsg x w t
  unfold Loop.linkOf linkMsg
  rcases hsx : FlockDeploy.setupOf true x with _ | ⟨s0, spec, sts⟩
  · rfl
  · have hst : stsOf x = sts := by
      unfold stsOf
      rw [hsx]
    have hn : jOf x = sts.size := by
      simp only [jOf, hx, ↓reduceIte, hst]
    simp only [Option.isNone_some, Bool.false_eq_true, ↓reduceIte]
    congr 2
    · rw [List.ofFn_congr hn.symm]
      congr 1
      funext j
      obtain ⟨l, hl, hi⟩ := hs ν t j
      rw [hst] at hl
      simp only [Fin.val_cast]
      rw [hl, ← hi, ← Loop.level0Of_root hl]
      rfl
    · rw [hn]

theorem linkRoundEq_ofLoop (vs : (session Loop.loopNr dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit ((session Loop.loopNr dataE).refP.Nonce × (session Loop.loopNr dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (session Loop.loopNr dataE).refP.Nonce × (session Loop.loopNr dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (session Loop.loopNr dataE).refP.Nonce × (session Loop.loopNr dataE).refP.Rand →
      ℕ → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray))
    (hL : L₀.link = fun x w ρ _ => Loop.linkOf src x w ρ) {x : Msg} (hx : Live x) {w : Flock.Wit}
    (hs : Loop.LinkLevel0 src x w) : LinkRoundEq (session Loop.loopNr dataE) vs (ofLoop dataE L₀ real) x w :=
  linkRoundEq_of_linkIs Loop.loopNr vs _ x w (linkIs_ofLoop L₀ src real hL hx hs)

theorem linkRoundEq_ofLoopLink (vs : (session Loop.loopNr dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit ((session Loop.loopNr dataE).refP.Nonce × (session Loop.loopNr dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (session Loop.loopNr dataE).refP.Nonce × (session Loop.loopNr dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (session Loop.loopNr dataE).refP.Nonce × (session Loop.loopNr dataE).refP.Rand →
      ℕ → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray))
    {x : Msg} {w : Flock.Wit} (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hs : Loop.LinkLevel0 src x w) :
    LinkRoundEq (session Loop.loopNr dataE) vs (ofLoopLink L₀ src real) x w :=
  linkRoundEq_ofLoop vs (withFiles (withLink L₀ src) src) src real rfl (live_of hsc hR) hs

end LeanRelease

end Claims.ZKServer
