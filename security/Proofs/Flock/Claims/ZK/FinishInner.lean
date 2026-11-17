module

public import Proofs.Flock.Claims.ZK.FinishPiop
public import Proofs.Flock.Claims.Complete.Zk.PadAlgebra
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
import Std.Tactic.BVDecide.Normalize.Bool

/-! The ZK server's inner proof is the masked session's where no constraint fails on the pads: the port's
`innerP` never checks the batched constraint, as `repFw` stops first where one fails (`failingOn`). -/

@[expose] public section

namespace Flock.ZkServer.Finish

open FlockComplete FlockLevel3 FlockSoundness.Discharge.Composed.ZerocheckMasked
open FlockSoundness.Discharge.Composed.Zerocheck (eq_of_of dot_eq_foldl of_foldl_dot)

def PadsZk.toC (p : PadsZk) : FlockComplete.PadsZk := ⟨p.h, p.mu, p.padH, p.padMu, p.tauSalt⟩

def InnerCommit.toC (ic : InnerCommit) : FlockComplete.InnerCommit := ⟨ic.ch, ic.cm, ic.salts, ic.layers, ic.root⟩

def Batched (lay : Zk.PadLayout) (h : Array F128) (cons : Array (String × Zk.Lin)) (eps : F128)
    (gammas : Array F128) : Bool :=
  let l := Zk.Lin.combo gammas ((cons ++ Zk.triple lay eps (eps * h[lay.fa]! + h[lay.tx]!)
    (h[lay.fb]! + h[lay.ty]!)).map (·.2)) lay.len
  (l.c + dot l.a h).isZero

section

variable {M : Type → Type} [Monad M]

theorem innerCommitP_eq [LawfulMonad M] (hs : Hashes M) (p : PadsZk) (salt : ℕ → ByteArray) :
    InnerCommit.toC <$> innerCommitP hs p salt = FlockComplete.innerCommitP hs p.toC salt := by
  simp only [FlockComplete.innerCommitP, innerCommitP, map_bind, map_pure]
  rfl

variable [MonadExceptOf String M] [MonadLiftT V M]

theorem innerP_eq (io : Asks M) (hs : Hashes M) (lay : Zk.PadLayout) (p : PadsZk) (ic : InnerCommit) (finalC : F128)
    (cons : Array (String × Zk.Lin)) (hb : ∀ eps gammas, Batched lay p.h cons eps gammas = true) :
    innerP io hs lay p ic finalC cons = FlockComplete.innerP io.toC hs lay p.toC ic.toC finalC cons := by
  simp only [Batched] at hb
  simp only [FlockComplete.innerP, innerP, PadsZk.toC, InnerCommit.toC, hb, Bool.not_true, Bool.false_eq_true,
    ↓reduceIte]
  rfl

end

theorem linAt_eq (l : Zk.Lin) (h : Array F128) : linAt l h = evalL l h := by
  apply eq_of_of
  rw [linAt, ← Array.foldl_toList, Array.toList_range, evalL, dot_eq_foldl, of_add, of_foldl_dot, of_foldl_dot,
    ← GF128.zero_eq, zero_add]

theorem consHold_of_failingOn {cons : Array (String × Zk.Lin)} {h : Array F128}
    (hs : ∀ q ∈ cons, q.2.a.size = h.size) (hf : failingOn cons h = #[]) : ConsHold cons h := by
  unfold failingOn at hf
  rw [Array.map_eq_empty_iff, Array.filter_eq_empty_iff] at hf
  intro q hq
  refine ⟨hs q hq, ?_⟩
  rw [← linAt_eq]
  simpa [bne] using hf q hq

theorem batched_of_failingOn (lay : Zk.PadLayout) {h : Array F128} {cons : Array (String × Zk.Lin)}
    (hs : h.size = lay.len) (hab : h[lay.hab]! = h[lay.fa]! * h[lay.fb]!) (htw : h[lay.tw]! = h[lay.tx]! * h[lay.ty]!)
    (hsz : ∀ q ∈ cons, q.2.a.size = h.size) (hf : failingOn cons h = #[]) (eps : F128) (gammas : Array F128) :
    Batched lay h cons eps gammas = true := by
  have e := batched_of_holds ((consHold_of_failingOn hsz hf).append (triple_holds lay h hs hab htw eps)) gammas
  rw [hs] at e
  unfold evalL at e
  simp only [Batched]
  rw [e]
  rfl

end Flock.ZkServer.Finish
