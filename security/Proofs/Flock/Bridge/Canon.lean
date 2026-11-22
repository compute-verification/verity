module

public import Proofs.Flock.Bridge.Statement
public import Proofs.Flock.EndToEndHidden.Statement
public import Proofs.Flock.Bridge.PortRoots
public import Proofs.Flock.Bridge.PopMatches
public import Proofs.Flock.Bridge.RightRegZc
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

/-!
# Canonical openings

A registered opening counts only when it is canonical at its wire (`Bridge.canonOpen`): the wire has a row
(`rows g = some r`), the opening's value is a row (`RowShaped`) of the row's width, and its salt is the `saltLen` bytes
the deployed leaf takes. The game side counts the units wrong with canonical openings (`incorrectRegZc`), at:

* `Bridge.canonRow rows g v`: `g` has a row `r`, `v` is `RowShaped` and `v`'s width is `r.width`;
* `Bridge.canonOpen rows g v y`: `canonRow rows g v`, and the salt `y` has `saltLen` bytes;
* `Bridge.canonAt rows κ g bt v y`: at a constant wire (`κ g = some b`) the opening's bit `bt` is `b`, at any other
  `canonOpen rows g v y`. A constant wire has no row: `progOf` reads it as `.const`, with no opening;
* `Bridge.FaultyRightTargetC`: M1's core over `RightRegZc`'s second conjunct, at canonical openings (a target, not proved).
-/

@[expose] public section

namespace Flock.Bridge

/-- **A canonical value at wire `g`**: `g` has a row `r`, and the value is a row (`RowShaped`) of `r`'s width. -/
def canonRow {N W : ℕ} (rows : Fin N → Option RowAt) (g : Fin N) (v : FlockSoundness.Discharge.Layout.BitVal W) :
    Prop :=
  ∃ r, rows g = some r ∧ FlockSoundness.Discharge.Layout.RowShaped v ∧ v.1.val = r.width

/-- **A canonical opening at wire `g`**: its value `v` is canonical at `g` (`canonRow`), and its salt `y` has the
`saltLen` bytes the deployed leaf takes (`saltOf`). -/
def canonOpen {N W : ℕ} (rows : Fin N → Option RowAt) (g : Fin N) (v : FlockSoundness.Discharge.Layout.BitVal W)
    (y : List UInt8) : Prop :=
  canonRow rows g v ∧ y.length = saltLen

/-- **A canonical opening at wire `g` at constants `κ`**: at a constant wire (`κ g = some b`) its bit `bt` on `g` is
`b`; at any other, it is `canonOpen`. -/
def canonAt {N W : ℕ} (rows : Fin N → Option RowAt) (κ : Fin N → Option Bool) (g : Fin N) (bt : Bool)
    (v : FlockSoundness.Discharge.Layout.BitVal W) (y : List UInt8) : Prop :=
  (∀ b, κ g = some b → bt = b) ∧ (κ g = none → canonOpen rows g v y)

/-- **The row a layout builds of `16 k` bits is canonical**: at width `K = 16 k ≤ W`, `(widthOf W K, padBits b)` is
`RowShaped` of width `K`. -/
theorem rowShaped_rowOf {W K k : ℕ} (hK : K = 16 * k) (hKW : K ≤ W) (b : Fin K → Bool) :
    FlockSoundness.Discharge.Layout.RowShaped
        ((FlockSoundness.Discharge.Layout.widthOf W K, (FlockSoundness.Discharge.Layout.padBits b : Fin W → Bool)) :
          FlockSoundness.Discharge.Layout.BitVal W) ∧
      (FlockSoundness.Discharge.Layout.widthOf W K).val = K := by
  have hw : (FlockSoundness.Discharge.Layout.widthOf W K).val = K := by
    simp only [FlockSoundness.Discharge.Layout.widthOf]; omega
  refine ⟨⟨by rw [hw, hK]; omega, fun i hi => ?_⟩, hw⟩
  change (FlockSoundness.Discharge.Layout.widthOf W K).val ≤ i.val at hi
  rw [hw] at hi
  simp only [FlockSoundness.Discharge.Layout.padBits, show ¬ i.val < K by omega, ↓reduceDIte]

/-- **M1's core, with canonical openings**: a unit whose committed wires each carry, in some canonical opening of their
registered commit string, their value in `C`'s evaluation on `Xr`'s inputs, and each of whose canonical registered
openings at a wire the constants `κ` don't fix opens its wire's position under `Cm` and `roots`, and whose evaluation is
`κ`'s constant at each constant wire it reads, is not faulty in `witDep C Cm H roots Xr` at the statement at `κ`, under
any design `r` and strata `ns` of the population, at constants that fix only input gates (`progOf` lowers only an input
to `.const`; a fixed computed wire would be neither a constant nor transported). -/
def FaultyRightTargetC : Prop :=
  ∀ {C : Verity.Boolean.Circuit} {n : ℕ} (P : Spot.Sampling.Partition C n) {Pos Val Opn Com : Type}
    (pos : Fin C.N → Pos) (bit : Val → Fin C.N → Bool) (val : Opn → Val) (com : Opn → Com) (cs : Pos → Com)
    (canon : Fin C.N → Val → Prop) (H : Hash.Msg → Hash.Msg) (Cm : Claims.Sampling.CommitmentScheme Bool)
    (roots : List Hash.Msg) (Xr : Fin C.N → Bool) (κ : Fin C.N → Option Bool) (u : Fin n),
    (∀ g b, κ g = some b → (C.op g).isInput = true) →
    (∃ op : Pos → Opn, ∀ g ∈ P.io u, com (op (pos g)) = cs (pos g) ∧ bit (val (op (pos g))) g = C.eval Xr g ∧
      canon g (val (op (pos g)))) →
    (∀ g ∈ P.io u, κ g = none → ∀ o : Opn, com o = cs (pos g) → canon g (val o) →
      ∃ d : Hash.Msg, Cm.Opens H roots ⟨posOf C g, bit (val o) g, d⟩ = true) →
    (∀ g ∈ P.io u, ∀ b, κ g = some b → C.eval Xr g = b) →
    ∀ (ns : List ℕ) (r : Claims.Sampling.OneStage) (desc : Hash.Msg) (q : Claims.Sampling.Query), ns.sum = n →
      u.val ∉ Claims.Sampling.faulty H Cm (stmtOf P κ r desc q roots) (partOf P κ ns) (witDep C Cm H roots Xr)

end Flock.Bridge

