module

public import Proofs.Flock.Claims.Recursion.BoundRouteCols
public import Proofs.Flock.Claims.Recursion.BoundWeight
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
import Proofs.Flock.Claims.Recursion.BoundBridge

/-! The deployed bound record at `coverW` (`bdCover`) and the route with (P3) discharged; no theorem evaluates
`Claims.Recursion.lam` at the deployment, at any weight. -/

@[expose] public section

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

/-- **`γ`'s weight at `IVStar` at the deployment**: (P3)'s floor at the deployed plain record's weight. -/
noncomputable def wCover (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) : Msg → ℝ :=
  coverW (FlockDeploy.deployPlainTrim rel).weight

/-- **The deployed bound record**: `bdDeployed` with `γ`'s weight at `IVStar` at `wCover`, at every other
choice the code side makes. -/
noncomputable def bdCover (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) :
    Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace :=
  bdDeployed rel nextE colsE wE join rowIn final (wCover rel)

def KSBoundCover (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) : Prop :=
  KSBoundDeployed rel relO b L nextE colsE wE join rowIn final (wCover rel)

theorem coverW_mono {wE wE' : Msg → ℝ} {x : Msg} (h : wE x ≤ wE' x) : coverW wE x ≤ coverW wE' x := by
  unfold coverW
  have hc : 0 ≤ (1 + 210 * (⌈4 * FlockDeploy.N₀ false x / Real.exp 1⌉₊ : ℝ)) *
      Real.sqrt ((FlockDeploy.V false).k x / 32) := by positivity
  exact pow_le_pow_left₀ (add_nonneg (Real.sqrt_nonneg _) hc) (add_le_add (Real.sqrt_le_sqrt h) le_rfl) 2

theorem weightCovers_bdCover (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (hwE : ∀ x, wE x ≤ (FlockDeploy.deployPlainTrim rel).weight x) :
    (bdCover rel nextE colsE wE join rowIn final).WeightCovers :=
  (weightCovers_deployed_iff rel nextE colsE wE join rowIn final (wCover rel)).2 fun x =>
    coverW_mono (wE' := (FlockDeploy.deployPlainTrim rel).weight) (hwE x)

theorem weightCovers_bdCover_plain (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) :
    (bdCover rel nextE colsE (FlockDeploy.deployPlainTrim rel).weight join rowIn final).WeightCovers :=
  weightCovers_coverW rel nextE colsE (FlockDeploy.deployPlainTrim rel).weight join rowIn final

/-- No theorem evaluates its level (`Claims.Recursion.lam`) at the deployment. -/
theorem ksBoundRouteCols_cover (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB)
    (hRows : L.ReadsRows FlockDeploy.sha512) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (hwE : ∀ x, wE x ≤ (FlockDeploy.deployPlainTrim rel).weight x) :
    let Bd := bdCover rel nextE colsE wE join rowIn final
    let D := recBound rel b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    Bd.PlainKSAbort FlockDeploy.sha512 → Bd.Binds → Bd.LastBindsEC FlockDeploy.sha512 →
    Bd.LastCoinFree FlockDeploy.sha512 →
    Claims.Flock.KS FlockDeploy.sha512 (deployZkRec relO) →
    (∃ decC : Msg → Msg → Coin,
      (∀ decI, Claims.Recursion.VStar.RunPairs FlockDeploy.sha512 D
        (Claims.Recursion.Inner.ofFlock Bd.deploy decI) decC) ∧
      ∀ shape ∈ [Claims.Recursion.Form.Dense, Claims.Recursion.Form.Holo, Claims.Recursion.Form.Sparse],
        ∃ (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
          (cmt : Msg → ℕ → Msg → Msg),
          (∀ decI, Claims.Recursion.VStarB.HeldB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape) ∧
          (∀ decI, Claims.Recursion.VStarB.VstarShowsB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape
            decC shows cmt) ∧
          (∀ s, Claims.Recursion.VStarB.PartChecksB FlockDeploy.sha512 D (deployZkRec relO) L shape shows cmt chk s) ∧
          (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec relO) L shape shows cmt s) ∧
          Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
    ReadersLogColl FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
    ReadersRunPlayed FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
    KSBoundCover rel relO b L nextE colsE wE join rowIn final := by
  intro _ _ _ hP1 hBinds hEC hCF hF hst hT hL hRRP
  exact ksBoundRoute_of_bridge_cols rel relO b L hRows (bridge_bdDeployed rel) nextE colsE wE join rowIn final
    (wCover rel) hP1 hBinds (weightCovers_bdCover rel nextE colsE wE join rowIn final hwE) hEC hCF hF hst hT hL hRRP

end RecDeploy
