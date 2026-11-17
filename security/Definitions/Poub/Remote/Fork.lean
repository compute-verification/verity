module

public import Definitions.Poub.Remote.Sample
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

/-! The receipts' fork game, which bounds their binding term; nothing here bounds `split`. -/

@[expose] public section

namespace ResourceAccounting.Remote.Sample

open scoped ENNReal

variable {n κ : ℕ} {X : Type}

namespace Receipts

variable (R : Receipts n κ X) {m : ℕ} {s : Fin n → ℕ}

def Broken (D : Run n m s κ X) (prf : Fin n → Pous.Bits κ → Blocks m) (rest : SampleCoins n κ X → Fin n → ℕ → ℕ)
    (t : Fin n) (ω ζ : SampleCoins n κ X) : Prop :=
  ∃ p ∈ D.opened prf rest t ω, p ∈ D.opened prf rest t (cont t ω ζ) ∧
    (Hash.CollisionIn R.hash
        (Hash.RowLeaf.outerHashed R.hash R.hash (R.verifier t ω p) (R.rank t ω p) (R.opened t ω p) ++
          Hash.RowLeaf.outerHashed R.hash R.hash (R.verifier t (cont t ω ζ) p) (R.rank t (cont t ω ζ) p)
            (R.opened t (cont t ω ζ) p)) ∨
      Hash.CollisionIn R.hash
        (Hash.RowLeaf.innerHashed R.hash (R.verifier t ω p) (R.opened t ω p) ++
          Hash.RowLeaf.innerHashed R.hash (R.verifier t (cont t ω ζ) p) (R.opened t (cont t ω ζ) p)))

/-- **Window `t`'s roots split** between `ω` and its continuation on `ζ`: some receipt the window opens in both has its
verifier, root or rank differ between them. -/
def Split (D : Run n m s κ X) (prf : Fin n → Pous.Bits κ → Blocks m) (rest : SampleCoins n κ X → Fin n → ℕ → ℕ)
    (t : Fin n) (ω ζ : SampleCoins n κ X) : Prop :=
  ∃ p ∈ D.opened prf rest t ω, p ∈ D.opened prf rest t (cont t ω ζ) ∧
    ¬ (R.verifier t (cont t ω ζ) p = R.verifier t ω p ∧ R.root t (cont t ω ζ) p = R.root t ω p ∧
      R.rank t (cont t ω ζ) p = R.rank t ω p)

noncomputable def fork [Fintype X] (D : Run n m s κ X) (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : SampleCoins n κ X → Fin n → ℕ → ℕ) (t : Fin n) : ℝ≥0∞ :=
  Pous.pr fun y : SampleCoins n κ X × SampleCoins n κ X => R.Broken D prf rest t y.1 y.2

/-- The probability that window `t`'s roots split over a uniform pair of coins. -/
noncomputable def split [Fintype X] (D : Run n m s κ X) (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : SampleCoins n κ X → Fin n → ℕ → ℕ) (t : Fin n) : ℝ≥0∞ :=
  Pous.pr fun y : SampleCoins n κ X × SampleCoins n κ X => R.Split D prf rest t y.1 y.2

/-- **What window `t`'s fork finder outputs on `(ω, ζ)`**: two different messages `uv` of one SHA-512 digest, both
among what the checks of the two openings of one receipt the window opens in both runs hashed. -/
def Finds (D : Run n m s κ X) (prf : Fin n → Pous.Bits κ → Blocks m) (rest : SampleCoins n κ X → Fin n → ℕ → ℕ)
    (t : Fin n) (ω ζ : SampleCoins n κ X) (uv : Hash.Msg × Hash.Msg) : Prop :=
  Claims.Collision (fun u => (R.hash u).bytes) uv.1 uv.2 ∧
    ∃ p ∈ D.opened prf rest t ω, p ∈ D.opened prf rest t (cont t ω ζ) ∧
      ((uv.1 ∈ Hash.RowLeaf.outerHashed R.hash R.hash (R.verifier t ω p) (R.rank t ω p) (R.opened t ω p) ++
            Hash.RowLeaf.outerHashed R.hash R.hash (R.verifier t (cont t ω ζ) p) (R.rank t (cont t ω ζ) p)
              (R.opened t (cont t ω ζ) p) ∧
          uv.2 ∈ Hash.RowLeaf.outerHashed R.hash R.hash (R.verifier t ω p) (R.rank t ω p) (R.opened t ω p) ++
            Hash.RowLeaf.outerHashed R.hash R.hash (R.verifier t (cont t ω ζ) p) (R.rank t (cont t ω ζ) p)
              (R.opened t (cont t ω ζ) p)) ∨
        (uv.1 ∈ Hash.RowLeaf.innerHashed R.hash (R.verifier t ω p) (R.opened t ω p) ++
            Hash.RowLeaf.innerHashed R.hash (R.verifier t (cont t ω ζ) p) (R.opened t (cont t ω ζ) p) ∧
          uv.2 ∈ Hash.RowLeaf.innerHashed R.hash (R.verifier t ω p) (R.opened t ω p) ++
            Hash.RowLeaf.innerHashed R.hash (R.verifier t (cont t ω ζ) p) (R.opened t (cont t ω ζ) p)))

open Classical in
/-- **Window `t`'s fork finder** at `q` SHA-512 calls: on coins `(ω, ζ)` it outputs what it finds (`Finds`), if
anything. -/
noncomputable def finder [Fintype X] [Nonempty X] (D : Run n m s κ X) (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : SampleCoins n κ X → Fin n → ℕ → ℕ) (t : Fin n) (q : ℕ) : Claims.Finder where
  S := SampleCoins n κ X × SampleCoins n κ X
  out y := if h : ∃ uv, R.Finds D prf rest t y.1 y.2 uv then some h.choose else none
  cost _ := q

end Receipts

end ResourceAccounting.Remote.Sample
