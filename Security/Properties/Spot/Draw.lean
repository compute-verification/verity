module

public import Definitions.Claims.Sampling
public import Spot.Slots
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
# Sampled proofs' sample in the claims' slots: it misses proof blocks no more often than `P_a` (`Spot.Slots.select`)

The property of the challenger's sample as the claims read it (`Deploy.checked`, `Claims.Sampling.DrawLaw`): the
sample `Slots.select` makes from `L` uniform bytes under a Spec's selection rule, over a circuit file and the verifier's
own program and query, checks no proof block at a slot of `B` with probability at most the claims' acceptance
probability `P_a(B)` of the design it samples, on the partition it samples over. It takes no premise: the sample is a
function of its inputs and its bytes, and every input is in it, the `L` bytes included, however few.

* **The partition** (`slotPartition`) is `Slots.fileLayout`'s: one class, the population in the registration's strata,
  each proof block its positions in the circuit file.
* **The design** (`slotDesign`) is the one the sample runs, one rule: the Flock law it samples (`Select.select`'s,
  `oneStageOf`), `all` for the `All` rule. The bound is `0` where the sample is refused before it samples (`missBound`).
* **The bytes** are `src`'s, in order (`bytesOf`): a sample that would read past them is refused, so it misses nothing.

`DesignWellFormed`: that design is well formed on that partition (the claims' `Design.WellFormed`), so its `P_a` is a
probability, wherever both are given; the sample's own refusals make it so, and it takes no premise either.
-/

@[expose] public section

open Lean (Json)
open Spot

namespace Properties.Spot

/-- `L` bytes as `Slots.select` reads its coins. -/
def bytesOf {L : ℕ} (src : Fin L → Fin 256) : ByteArray := ⟨Array.ofFn fun i => UInt8.ofFin (src i)⟩

/-- A layout's proof block, its positions' intervals `[lo, hi)` in order, as the claims' set of gates. -/
def puGates (b : Slots.ProofBlock) : List ℕ := b.gates.toList.flatMap fun r => List.range' r.1 (r.2 - r.1)

/-- The claims' partition of a layout: its classes, each its strata, each its proof blocks' gates. -/
def slotPartition (l : Slots.Layout) : Claims.Sampling.Partition :=
  ⟨l.classes.toList.map fun c => c.toList.map fun s => s.toList.map puGates⟩

/-- The claims' one-stage design of a Flock law. -/
def oneStageOf : Flock.Draw.Law → Claims.Sampling.OneStage
  | .subset k => .subset k
  | .bernoulli num den => .bernoulli num den
  | .stratified k _ => .stratified k
  | .work k ws => .work k (ws.toList.map (·.work)) (ws.toList.map (·.floor))

/-- **The design `Slots.select` samples** under `rule` on `d` and `query` with the work `table`: one rule, the
sampling design `Select.select` samples (`all` for the `All` rule). An error where `Select.select` refuses before it
samples. -/
def slotDesign (rule : Select.Rule) (d : Program) (query : Json) (table : Option Json) : Flock.V Claims.Sampling.Design := do
  let n ← Partition.units d query
  let named ← rule.lawOver n
  let derived ← Draw.deriveLaw d query named table
  let first ← match rule with
    | .all => pure Claims.Sampling.OneStage.all
    | _ => do
      let parsed ← Draw.flockLaw ((derived.map (·.toJson)).getD named) n
      pure (oneStageOf (derived.getD parsed))
  return ⟨[first]⟩

/-- The bound: `P_a(B)` of the design on the partition the sample runs over, and `0` where the sample is refused before it
samples. -/
noncomputable def missBound (rule : Select.Rule) (c : Flock.Circuit) (d : Program) (query : Json) (table : Option Json)
    (B : Finset ℕ) : ℝ :=
  match Slots.fileLayout c d query, slotDesign rule d query table with
  | .ok l, .ok D => D.Pa (slotPartition l) B
  | _, _ => 0

/-- The slots a sample checks, none where it is refused. -/
def checkedOf : Flock.V (Slots.Layout × Array ℕ) → Array ℕ
  | .ok (_, c) => c
  | .error _ => #[]

/-- The checked slots `c` miss `B`: none is a slot of `B`. -/
def Misses (B : Finset ℕ) (c : Array ℕ) : Prop := ∀ v ∈ B, v ∉ c

/-- **The sample misses `B` with probability at most `P_a(B)`**: for every selection rule, circuit file, program, query,
work table, closure map and number `L` of uniform bytes, and every set `B` of slots, the sample `Slots.select` makes from
the bytes is accepted and checks no proof block at a slot of `B` with probability at most `missBound`. -/
def DrawMisses : Prop :=
  ∀ (rule : Select.Rule) (c : Flock.Circuit) (d : Program) (query : Json) (table : Option Json)
    (cm : Option (Array (Int × Array Int))) (L : ℕ) (B : Finset ℕ),
    Claims.Pr (fun src : Fin L → Fin 256 =>
      (Slots.select rule c d query table cm (bytesOf src)).isOk = true ∧
        Misses B (checkedOf (Slots.select rule c d query table cm (bytesOf src)))) ≤
      missBound rule c d query table B

/-- **The design the sample runs is well formed on the partition it runs over**: wherever `Slots.fileLayout` lays out
the circuit file `c` in the strata of `d` under `query` and `slotDesign` gives the design, it is the claims'
`Design.WellFormed` there. One rule for the one class, `subset K` with `1 ≤ K ≤ N`, Bernoulli at `NUM/DEN` with
`0 < DEN` and `NUM ≤ DEN`, `stratified K` with `1 ≤ K ≤ N` over strata of at least one unit each, or `work K` over such
strata with one work and one floor of at least one per stratum, and some work. -/
def DesignWellFormed : Prop :=
  ∀ (rule : Select.Rule) (c : Flock.Circuit) (d : Program) (query : Json) (table : Option Json) (l : Slots.Layout)
    (D : Claims.Sampling.Design),
    Slots.fileLayout c d query = .ok l → slotDesign rule d query table = .ok D → D.WellFormed (slotPartition l)

end Properties.Spot
