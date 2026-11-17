module

public import Proofs.Flock.Claims.Recursion.BitsLin2
public import Proofs.Flock.Claims.Recursion.CapPorts
public import Proofs.Flock.Claims.Recursion.KSLin2Staged
public import Proofs.Flock.Claims.Recursion.LayoutBTops
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
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Recursion.CallsOfProofs
import Proofs.Flock.Claims.Recursion.DeployFacts
import Proofs.Flock.Claims.Recursion.LastBindsClosed
import Proofs.Flock.Claims.Recursion.RowsCover
import Proofs.Flock.Claims.Recursion.SchedDense

/-! `KSLin2` and `BitsLin2` at `layoutBTops` with V\*'s staging read off the run, since `ksLin2_deployed_caps`'s `VstarShowsB` holds only where no honest run is accepted. -/

@[expose] public section

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin)

theorem heldWhole_layoutBTops (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) :
    (layoutBTops fin).HeldWhole := by
  intro x s i hi hs
  have hne : i + 1 ≠ layoutOf.rounds x := Nat.ne_of_lt hi
  change stepPieces fin x i = []
  unfold stepPieces
  rcases i with _ | j
  · simp [hne]
  · simp only [hne, ↓reduceIte, Nat.add_one_ne_zero, Nat.add_sub_cancel]
    unfold topsPieces
    simp only [layoutBTops, streamRound] at hs
    rcases hr : readX x with _ | r
    · rfl
    simp only [hr, Option.bind_some] at hs ⊢
    rcases hrs : repSched r.shape with _ | rs
    · rfl
    simp only [hrs, Option.bind_some, Option.map_some, Option.getD_some] at hs ⊢
    unfold topsPiecesAt
    by_cases ht : (rs.getD (j % rs.size) default).2.1.isEmpty = true
    · simp only [ht, ↓reduceIte]
    · exfalso
      by_cases hj : j < 2 * rs.size
      · simp only [hj, ↓reduceIte, Option.bind_some, ht, Option.isSome_none, Bool.false_eq_true] at hs
      · simp only [hj, ↓reduceIte, Option.bind_none, Option.isSome_none, Bool.false_eq_true] at hs

/-- This bounds no number by itself: `FinPieces`, `PartChecksBV`, `PartOpensB`, `DecompB`, `ReadersTrue` and `ReadsBackB` stay open. -/
theorem ksLin2_deployed_staged (b : Brings) (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (untr : TrReader) :
    let L := layoutBTops fin
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    ReadsBackB FlockDeploy.sha512 Bd.IVStar bL (fun _ => True) untr →
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      FinPieces fin shape ∧
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart untr b) (cmtTr untr) chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart untr b) (cmtTr untr) s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro L Bd D bL hrb hP hRT
  refine ksLin2_deployed_plain_parts b L (Claims.Recursion.VStarB.LayoutB.readsRows_of_rowIn fun _ _ => rfl)
    (heldWhole_layoutBTops fin) wE join final w canonicalProofFiles plainCallsBindOrSlack_sha512 untr hrb ?_ hRT
  intro shape hsh
  obtain rfl := List.mem_singleton.1 hsh
  obtain ⟨hF, chk, hPC, hPO, hD⟩ := hP _ hsh
  exact ⟨chk, fun decI => heldB_layoutBTops _ _ _ _ (fun _ => rfl) fin _ schedMatches_dense (capPorts _)
    (rowsCover _) hF, hPC, hPO, hD⟩

/-- A bound on a number under `StrictBirthday`, `ExpectedCR` and `ksLin2_deployed_staged`'s open premises, not a proved security level. -/
theorem bitsLin2_deployed_staged (b : Brings) (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (untr : TrReader) :
    let L := layoutBTops fin
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    ReadsBackB FlockDeploy.sha512 Bd.IVStar bL (fun _ => True) untr →
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      FinPieces fin shape ∧
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart untr b) (cmtTr untr) chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart untr b) (cmtTr untr) s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.BitsLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro L Bd D bL hrb hP hRT
  exact Claims.Recursion.bitsLin2On_of_ksLin2On
    (sessionAccepts Bd.IVStar (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck bL)
    (ksLin2_deployed_staged b fin wE join final w untr hrb hP hRT)

end RecDeploy
