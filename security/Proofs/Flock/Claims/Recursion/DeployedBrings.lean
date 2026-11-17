module

public import Proofs.Flock.Claims.Recursion.RegSetup
public import Proofs.Flock.Claims.Recursion.ResLowerDeployed
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

/-! The deployed brings' setup `deployedB`, the compiled verifier's own SHA-512 evaluations at a V\* statement, read off
the statement alone; it leaves out rows no instance references, which no conclusion reads. -/

@[expose] public section

namespace RecDeploy

/-- **The compiled verifier's own SHA-512 evaluations at the V\* statement `xO`**: check 5's (`ResLower.resSetup`),
then the registered reads' leaves and climbs (`regIns`). -/
def deployedSetup (xO : Claims.Msg) : List Claims.Msg := ResLower.resSetup xO ++ regIns xO

/-- **The deployed brings** at `b`: `b` with the verifier's own evaluations as its setup. -/
def deployedB (b : Brings) : Brings := { b with setup := deployedSetup }

theorem deployedB_setup (b : Brings) : (deployedB b).setup = deployedSetup := rfl

theorem resSetup_deployedB (b : Brings) : ∀ xO, ∀ u ∈ ResLower.resSetup xO, u ∈ (deployedB b).setup xO :=
  fun _ _ hu => List.mem_append_left _ hu

theorem regSetup_deployedB (b : Brings) : RegSetup (deployedB b) := fun _ _ hu => List.mem_append_right _ hu

theorem regListed_deployedB (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (IV : Claims.Verifier FlockDeploy.Coin) :
    RegListed (recBound rel (deployedB b) L IV) :=
  regListed_recBound rel (regSetup_deployedB b) L IV

theorem deployedB_tr (b : Brings) (t : Claims.Msg → List (Claims.Msg × Claims.Out) → ByteArray) :
    { deployedB b with tr := t } = deployedB { b with tr := t } := rfl

end RecDeploy
