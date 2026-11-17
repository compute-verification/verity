module

public import Definitions.Compose.Compose
public import Properties.Spot.Draw
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

/-! `UnitsOfLayout`: a registered circuit's proof blocks, commit coins, seed preimages and sample are the service's;
it is stated, not proved, because nothing builds a `Declared` from a compiled Program yet. -/

@[expose] public section

open Lean (Json)
open Pouw.RelativeSecurity (prob)

namespace Properties.Compose

def UnitsOfLayout {ι : Type} (Cd : ResourceAccounting.Compose.Declared ι) (Np Kd : ℕ) (p : Spot.Program)
    (query : Json) : Prop :=
  ∀ l, Spot.Slots.layout p query = .ok l →
    (∀ i, i < Cd.dag.length → Cd.binding i = none →
      ∃ b ∈ l.blocks.toList, b.unit = some (Cd.unit i) ∧ b.gates.1 ≤ i ∧ i < b.gates.2) ∧
    (∀ b ∈ l.blocks.toList, b.unit = none → b.gates.1 < Cd.dag.length ∧ (Cd.binding b.gates.1).isSome) ∧
    (∀ b ∈ l.blocks.toList, ∀ u, b.unit = some u → u < Cd.units) ∧
    (∀ ℓ, Cd.Writes ℓ → Cd.commitRound ℓ = match Cd.binding ℓ with
      | some .registered => 0
      | _ => Cd.rounds) ∧
    (∀ r ℓ, Cd.Writes ℓ → (Cd.seedNames r ℓ = true ↔ Cd.binding ℓ = some .registered)) ∧
    (Spot.Partition.units p query = .ok Np →
      ∀ (cm : Option (Array (Int × Array Int))) (L : ℕ) (U : Finset ℕ),
        prob (fun src : Fin L → Fin 256 => ∃ r,
            Spot.Select.select (.samplingDesign ("subset:" ++ toString Kd)) p query none cm
              (Properties.Spot.bytesOf src) = .ok r ∧ ∀ u ∈ U, u ∉ r.2.1) ≤
          prob (fun β : Equiv.Perm (Fin Np) => ∀ t ∈ Pouw.Window.subsetDraw Kd β, t.val ∉ U))

end Properties.Compose
