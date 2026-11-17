module

public import Proofs.Flock.Claims.Recursion.OpenPieces
public import Proofs.Flock.Claims.Recursion.VerdictRowD
public import Proofs.Flock.Claims.Recursion.VerdictRows
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

/-! Row D of the verdict from the `rec-open` pieces (`DecAt.rowD_of_pieces`). -/

@[expose] public section

namespace RecDeploy.Rows

open Flock
open Flock.RecAlgebra (verifierRows)
open FlockSoundness.Refine (shapeOf)
open FlockSoundness.Discharge.Recursion (HClass)

namespace DecAt

variable {rd : Claims.Msg → Option FlockDeploy.FinalP} {x : Claims.Msg} {ms : List Claims.Msg}
  {cs : List FlockDeploy.Coin} (D : DecAt rd x ms cs) (cls : HClass) (acc : ℕ → Array F128) (i : ℕ) (r : Fin 2)

def side : OpenSide :=
  ⟨(D.files cls i).sch, ((D.files cls i).proof r).cap, ((D.files cls i).proof r).ligerito⟩

theorem rowD_of_pieces
    (hmerkle : (D.files cls i).st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted")
    (hfast : Flock.fast100 (D.files cls i).st.m = some (D.files cls i).sch)
    (hdec : decodeProof (shapeOf (D.files cls i).st (D.files cls i).sch (D.files cls i).extras.size)
      ((D.files cls i).bytes r) = .ok ((D.files cls i).proof r))
    (pieces : List OpenPiece) (sideAt : Claims.Msg → List Claims.Msg → OpenPiece → OpenSide)
    (posAt : Claims.Msg → List FlockDeploy.Coin → OpenPiece → ℕ)
    (hholds : ∀ q ∈ pieces, openHolds sideAt posAt q x ms cs)
    (hside : ∀ q ∈ pieces, q.table = i → q.rep = r.val → sideAt x ms q = D.side cls i r)
    (hpos : ∀ v info kappa,
      verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
        .ok (v, info, kappa) →
      ∀ q ∈ pieces, q.table = i → q.rep = r.val → posAt x cs q = (info.lg.conss[q.level]!).pos[q.query]!)
    (hcover : ∀ ℓ < (D.files cls i).sch.levels.size, ∀ j < ((D.files cls i).sch.levels[ℓ]!).q,
      ∃ q ∈ pieces, q.table = i ∧ q.rep = r.val ∧ q.level = ℓ ∧ q.query = j) :
    D.RowD cls acc i r :=
  D.rowD_of_queries cls acc i r hmerkle hfast hdec fun v info kappa hv ℓ hℓ j hj => by
    obtain ⟨q, hq, hi, hr, rfl, rfl⟩ := hcover ℓ hℓ j hj
    have h := hholds q hq
    unfold openHolds at h
    rw [hside q hq hi hr, hpos v info kappa hv q hq hi hr] at h
    exact h

end DecAt

end RecDeploy.Rows
