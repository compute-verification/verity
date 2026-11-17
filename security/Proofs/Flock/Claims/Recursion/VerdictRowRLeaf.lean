module

public import Proofs.Flock.Claims.Recursion.ElemRead
public import Proofs.Flock.Claims.Recursion.VerdictRowRDep
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

/-! Row R's leaf at the concrete readers (`targetRDep_readers`), which still takes that `resP x` has the pieces (`ResPiecesAt`). -/

@[expose] public section

namespace RecDeploy.DecompBR

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP)
open Flock.RecAlgebra (structureOf)
open Flock.RecResiduals (portSizes groups)
open FlockSoundness.Discharge.Recursion (HClass)
open RecDeploy.Rows (DecAt elemReadC resVRead)

def ResPiecesAt (rd : Msg → Option FinalP) (resP : Msg → List ResPiece) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (side : Msg → List Msg → OpenPiece → OpenSide) (posAt : Msg → List Coin → OpenPiece → ℕ) : Prop :=
  ∀ (x : Msg) (ms : List Msg) (cs : List Coin),
    Top.At rd (PD rd resP key (elemReadC (Top.accRC rd key)) resVRead side posAt) x ms cs →
    ∀ (D : DecAt rd x ms cs) (cls : HClass) (i : ℕ), i < D.sts.size →
    ∀ (S : RecAlgebra.GfS) (gs : Array (Array ℕ × Array ℕ × Array ℕ)),
      structureOf (D.sh cls i) = .ok S → groups S (portSizes (D.sh cls i)) = .ok gs →
      ∀ (r : Fin 2) p (_ : p < gs.size), ∃ q ∈ resP x,
        (∃ ns nv, RecResiduals.part (D.sh cls i) p = .ok (q.S, ns, nv)) ∧
        q.key.1 = p ∧ q.key.2.1 + q.key.2.2.1 = r.val

theorem targetRDep_readers (rd : Msg → Option FinalP) {resP : Msg → List ResPiece}
    {key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ} {side : Msg → List Msg → OpenPiece → OpenSide}
    {posAt : Msg → List Coin → OpenPiece → ℕ} (hp : ResPiecesAt rd resP key side posAt) :
    Top.TargetRDep rd (PD rd resP key (elemReadC (Top.accRC rd key)) resVRead side posAt) (Top.accRC rd key) :=
  targetRDep_accRC rd fun x ms cs hA D cls i hi S gs hS hg r p hpp => by
    obtain ⟨⟨xo, sto, hst, hf, hx⟩, -⟩ := id hA
    obtain ⟨q, hq, hpart, h1, hog⟩ := hp x ms cs hA D cls i hi S gs hS hg r p hpp
    exact ⟨q, hq, hpart, h1, hog, fun k hk => D.elemReadC_eq cls (Top.accRC rd key) hst hf hx hi hS hg r hpp q.key h1
      hog k hk⟩

end RecDeploy.DecompBR
