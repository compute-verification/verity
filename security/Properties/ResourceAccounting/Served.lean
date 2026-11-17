module

public import Definitions.ResourceAccounting.Served
public import Properties.Compose.Compose
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

/-! The served window's units and sample are the service's, per call: stated, not proved. -/

@[expose] public section

open Lean (Json)
open Pouw.RelativeSecurity (prob)

namespace Properties.Served

open ResourceAccounting.Compose ResourceAccounting.Served

/-- **Call site `c`'s gates are its Program's**, on layout `l`: each non-input gate of the call lies in a proof block
of `l`, its declared unit that block's number shifted by the call's base; each input slot's gate is an input; and
every block's number is below the call's units. -/
def CallUnits {ι : Type} (L : Shape) (Cd : Declared ι) (start : ℕ → ℕ) (c : ℕ) (l : Spot.Slots.SlotOrder) :
    Prop :=
  (∀ i, start c ≤ i → i < start (c + 1) → i < Cd.dag.length → Cd.binding i = none →
    ∃ b ∈ l.blocks.toList, ∃ u, b.unit = some u ∧ Cd.unit i = L.base (L.stepOf c) (L.linearOf c) + u ∧
      b.gates.1 ≤ i - start c ∧ i - start c < b.gates.2) ∧
    (∀ b ∈ l.blocks.toList, b.unit = none →
      start c + b.gates.1 < start (c + 1) ∧ (Cd.binding (start c + b.gates.1)).isSome) ∧
    (∀ b ∈ l.blocks.toList, ∀ u, b.unit = some u → u < L.unitsOf (L.linearOf c))

/-- **The registered circuit's units are the served window's, call by call**: linear `w`'s Program has `unitsOf w`
units, the registered circuit the window's `population`, each gate lies in its call site's range, and every call
site's gates are its Program's (`CallUnits`). -/
def ServedUnits {ι : Type} (L : Shape) (Cd : Declared ι) (prog : ℕ → Spot.Program) (query : ℕ → Json)
    (start : ℕ → ℕ) : Prop :=
  (∀ w, w < L.calls → Spot.Partition.units (prog w) (query w) = .ok (L.unitsOf w)) ∧
    Cd.units = L.population ∧
    (∀ i, i < Cd.dag.length → start (Cd.call i) ≤ i ∧ i < start (Cd.call i + 1)) ∧
    ∀ c, c < Cd.calls → ∀ l, Spot.Slots.layout (prog (L.linearOf c)) (query (L.linearOf c)) = .ok l →
      CallUnits L Cd start c l

/-- **The verifier's subset sample over a population misses no more often than a uniform `K`-subset**: for every `K ≤ N`,
number `L` of uniform bytes and set `U`, `sampleUnits (subset K) N` is accepted on the bytes and samples no unit of `U` with
probability at most the probability that `Pouw.Window.subsetDraw K` of a uniform permutation of `Fin N` misses `U`. -/
def SampleMissesSubset : Prop :=
  ∀ (K N L : ℕ) (U : Finset ℕ), K ≤ N →
    prob (fun src : Fin L → Fin 256 => ∃ r,
        Spot.Draw.sampleUnits (.subset K) N (Properties.Spot.bytesOf src) = .ok r ∧
          ∀ u ∈ U, u ∉ r.1) ≤
      prob (fun β : Equiv.Perm (Fin N) => ∀ t ∈ Pouw.Window.subsetDraw K β, t.val ∉ U)

/-- **`UnitsOfLayout`'s seeded schedule and `LayerCommitted` exclude each other** on a well-formed registered circuit
with a run input some gate reads, so the live statements can't take `UnitsOfLayout` as written. -/
def SeededScheduleNotLive : Prop :=
  ∀ (ι : Type) (Cd : Declared ι) (Np Kd : ℕ) (p : Spot.Program) (query : Json)
    (l : Spot.Slots.SlotOrder),
    Spot.Slots.layout p query = .ok l → Properties.Compose.UnitsOfLayout Cd Np Kd p query → Cd.Wf →
      (∃ ℓ j, Cd.binding ℓ = some .run ∧ j < Cd.dag.length ∧ Cd.Reads j ℓ) → ¬ LayerCommitted Cd

/-- **The served hypothesis**: registered circuit `Cd` is served window `L` in form `F` (`Tie`), its units are the
per-call Programs' (`ServedUnits`), and the verifier's sample misses as a uniform subset does
(`SampleMissesSubset`). -/
def IsServed {ι : Type} (L : Shape) (F : Form) (Cd : Declared ι) (cls : ℕ → Cls) (prog : ℕ → Spot.Program)
    (query : ℕ → Json) (start : ℕ → ℕ) : Prop :=
  Tie L F Cd cls ∧ ServedUnits L Cd prog query start ∧ SampleMissesSubset

/-- **A tied circuit is in Compose's scope given the three conjuncts the tie doesn't read**: `Tie` gives `Scope`'s
`Declared.Live` (`TieLive`) and `LayerCommitted` (`TieCommits`), so a tied circuit that is well formed, with a positive
`δs` and its scheme's luck proved, meets `Scope`. -/
def TieScope : Prop :=
  ∀ (ι K : Type) (Sc : Scheme K) (L : Shape) (F : Form) (Cd : Declared ι) (cls : ℕ → Cls) (δs : ℝ),
    Tie L F Cd cls → Cd.Wf → 0 < δs → Sc.LuckProved → Scope Sc Cd δs

end Properties.Served
