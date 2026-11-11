module

public import Proofs.Flock.Claims.Recursion.VerdictRows
public import Proofs.Flock.VBridge.AlgResZero
public import Proofs.Flock.VBridge.Family.Join
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
import Proofs.Flock.VBridge.Algebra.BitsOk

/-!
# Row R: the residuals at a rep of the decision

`RowR` asks `AlgebraHolds` at rep `r` of table `i` (`DecAt.rep`). `rowR_of_parts` proves it from what the verdict's
`rec-residuals` pieces give, one fact per part of the verifier's algebra at the rep's shape (`PartsZero`): the part's
residuals vanish (VerdictPieces' `ResZero`) at elements `E` that are the rep's `w` at the part's row ports and verifier
values `vv` that are `verifierRows`' `v` at the part's `V`. The last is check 5's tie at the verifier's values, which
stays a hypothesis, as in `ResTie`.

The rest of `AlgebraHolds` is the shape's: S and its groups (`structureOf`, `groups`), the rep's coins in the rounds
`verifierRows` samples (`roundSizes`, with none empty, row B's `noEmpty`) and `w`'s size S's `nw`. The extra claims number
the shape's by definition (`shapeOf`), the comb has 64 elements (`combOf_size`), so `v` has S's `nv`
(`verifierRows_v_size`) and the wire bits are in range (`structureOf_bitsOk`).
-/

@[expose] public section

namespace RecDeploy.Rows

open Flock
open FlockDeploy (Coin FinalP)
open FlockVBridge (resVal termVal gfsOf resOf)
open FlockVBridge.Family (combOf_size verifierRows_v_size roundSizes roundSizes_spec)
open Flock.RecResiduals (portSizes groups)
open Flock.RecAlgebra (structureOf verifierRows el NoEmpty)
open FlockVBridge.PartS (wIdx)
open FlockSoundness.Discharge.Recursion (HClass)

namespace DecAt

variable {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
  (D : DecAt rd x ms cs) (cls : HClass) (acc : ℕ → Array F128) (i : ℕ) (r : Fin 2)

/-- **Every part's residuals vanish at rep `r`'s values**: for each group `p` of S, the verifier's part `p` has every
residual zero (`ResZero`) at elements `E` that are the rep's `w` at `wIdx` of the group's ports and values `vv` that
are, wherever `verifierRows` returns at the rep's coins, extra claims and comb, its `v` at the group's `V` (check 5's
tie). -/
def PartsZero (gs : Array (Array ℕ × Array ℕ × Array ℕ)) : Prop :=
  ∀ p (hp : p < gs.size), ∃ S' ns nv, RecResiduals.part (D.sh cls i) p = .ok (S', ns, nv) ∧
    ∃ E vv : ℕ → F128,
      (∀ k (hk : k < (wIdx (portSizes (D.sh cls i)) gs[p].2.1).size),
        E k = el (D.rep cls acc i r).w (wIdx (portSizes (D.sh cls i)) gs[p].2.1)[k]) ∧
      (∀ v info kappa, verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras
          (D.rep cls acc i r).comb = .ok (v, info, kappa) →
        ∀ k (hk : k < gs[p].2.2.size), vv k = el v gs[p].2.2[k]) ∧
      ∀ q (hq : q < S'.res.length), resVal (gfsOf S') (termVal E vv) (resOf S'.res[q]) = 0

/-- **`verifierRows` returns at rep `r`** when S does, none of the rep's coin rounds is empty and they have the sizes
the shape's schedule samples: the extra claims number the shape's by definition (`shapeOf`). -/
theorem verifierRows_rep {S : RecAlgebra.GfS} (hS : structureOf (D.sh cls i) = .ok S)
    (hne : NoEmpty (D.rep cls acc i r).rounds)
    (hsz : (D.rep cls acc i r).rounds.toList.map Array.size = roundSizes (D.sh cls i)) :
    ∃ y, verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras
      (D.rep cls acc i r).comb = .ok y :=
  (roundSizes_spec hS hne _ _).2 ⟨hsz, rfl⟩

/-- **Row R from the parts' residuals**: at S and its groups, with rep `r`'s coin rounds none empty and of the
schedule's sizes and its `w` of S's `nw` elements, every part's residuals vanishing at the rep's values (`PartsZero`)
gives `RowR`. -/
theorem rowR_of_parts {S : RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hS : structureOf (D.sh cls i) = .ok S) (hg : groups S (portSizes (D.sh cls i)) = .ok gs)
    (hne : NoEmpty (D.rep cls acc i r).rounds)
    (hsz : (D.rep cls acc i r).rounds.toList.map Array.size = roundSizes (D.sh cls i))
    (hw : (D.rep cls acc i r).w.size = S.nw) (hparts : D.PartsZero cls acc i r gs) :
    D.RowR cls acc i r := by
  obtain ⟨⟨v, info, kappa⟩, hvr⟩ := D.verifierRows_rep cls acc i r hS hne hsz
  have hc : (D.rep cls acc i r).comb.size = 64 := combOf_size _ _ _
  refine FlockVBridge.OfVStar.algebraHolds_of_resZero hS (Flock.RecAlgebra.structureOf_bitsOk hS) hvr hw hc
    (verifierRows_v_size hS hvr hc) hg fun p hp => ?_
  obtain ⟨S', ns, nv, hpart, E, vv, hE, hvv, h0⟩ := hparts p hp
  exact ⟨S', ns, nv, hpart, E, vv, hE, hvv v info kappa hvr, h0⟩

end DecAt

end RecDeploy.Rows
