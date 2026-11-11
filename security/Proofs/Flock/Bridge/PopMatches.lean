module

public import Proofs.Flock.Bridge.Defs
public import Proofs.Flock.Claims.Sampling.Reference.DeployedCode
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
# The registration's population is the game's units, in order (`Bridge.PopMatches`)

`Bridge.PopMatches desc q n`, a decidable check on the files, as `Bridge.RowsOk` is: the descriptor bytes `desc` parse
(`Spot.Program.parse`), the layout the deployed verifier evaluates on them under the registration's query `q`
(`Spot.Slots.layout`, which reads the query as JSON, `Reg.query`) is defined, and its population, the layout's
first class, is `n` proof blocks with slot `u` unit `u`. It says nothing of the units' gates: that slot `u`'s gates are
unit `u`'s lowered gates needs the descriptor-to-circuit-file lowering, which is not yet stated.

* `popOf l`: a layout's population, its first class's proof blocks in slot order.
* `popMatches_iff`: `PopMatches` is that the parse and the layout succeed and the population's units are
  `0, …, n - 1`, in order.
* `popOf_size`, `popOf_unit`: at `PopMatches`, the population has `n` slots, and slot `u` is unit `u`.
* `nsOf desc q`: the sizes of the population's strata, one per template in the order of its first unit, at the layout
  `PopMatches` reads; `nsOf_sum`: at `PopMatches`, they sum to `n`.
* `popCount Pt`: the number of proof blocks in a claims partition's first class, its population.
* `popCount_slotPartition`: the claims' partition of a layout (`slotPartition`) has `popOf`'s slots as its population.
* `popCount_partOf`: at `PopMatches`, that population has as many slots as the game's partition `partOf P κ ns`'s
  population (`popCount_partOf_eq`): N = n.
* `popCount_regMatches`: where `RegMatches` names a statement's registration and `PopMatches` holds at the statement's
  descriptor and the registration's query, the statement's partition has `popCount (partOf P κ ns)` population slots.
-/

@[expose] public section

namespace Flock.Bridge

open Properties.Spot

/-- **A layout's population**: its first class's proof blocks, in slot order. -/
def popOf (l : Spot.Slots.Layout) : List Spot.Slots.ProofBlock :=
  ((l.classes.toList.headD #[]).toList.map Array.toList).flatten

/-- `PopMatches`'s check. -/
def popCheck (desc : Claims.Msg) (q : Lean.Json) (n : ℕ) : Bool :=
  match Spot.Program.parse ⟨desc.toArray⟩ with
  | .error _ => false
  | .ok p =>
    match Spot.Slots.layout p q with
    | .error _ => false
    | .ok l => decide ((popOf l).map (·.unit) = (List.range n).map some)

/-- **The registration's population is `n` units, in order**: `desc` parses, its layout under the query `q` is
defined, and the layout's population is `n` proof blocks with slot `u` unit `u`. -/
@[reducible] def PopMatches (desc : Claims.Msg) (q : Lean.Json) (n : ℕ) : Prop := popCheck desc q n = true

instance instDecidablePopMatches (desc : Claims.Msg) (q : Lean.Json) (n : ℕ) : Decidable (PopMatches desc q n) :=
  inferInstanceAs (Decidable (_ = true))

theorem popMatches_iff {desc : Claims.Msg} {q : Lean.Json} {n : ℕ} :
    PopMatches desc q n ↔ ∃ p l, Spot.Program.parse ⟨desc.toArray⟩ = .ok p ∧
      Spot.Slots.layout p q = .ok l ∧ (popOf l).map (·.unit) = (List.range n).map some := by
  unfold PopMatches popCheck
  constructor
  · intro h
    split at h
    · exact absurd h Bool.false_ne_true
    · rename_i p hp
      split at h
      · exact absurd h Bool.false_ne_true
      · rename_i l hl
        exact ⟨p, l, hp, hl, of_decide_eq_true h⟩
  · rintro ⟨p, l, hp, hl, h⟩
    rw [hp]
    show (match Spot.Slots.layout p q with
      | .error _ => false
      | .ok l => decide ((popOf l).map (·.unit) = (List.range n).map some)) = true
    rw [hl]
    exact decide_eq_true h

/-- The population's units at the layout `PopMatches` reads. -/
theorem popOf_units {desc : Claims.Msg} {q : Lean.Json} {n : ℕ} {p : Spot.Program}
    {l : Spot.Slots.Layout} (h : PopMatches desc q n)
    (hp : Spot.Program.parse ⟨desc.toArray⟩ = .ok p) (hl : Spot.Slots.layout p q = .ok l) :
    (popOf l).map (·.unit) = (List.range n).map some := by
  obtain ⟨p', l', hp', hl', hu⟩ := popMatches_iff.1 h
  rw [hp] at hp'
  cases hp'
  rw [hl] at hl'
  cases hl'
  exact hu

/-- **N = n at the layout**: at `PopMatches`, the population has `n` slots. -/
theorem popOf_size {desc : Claims.Msg} {q : Lean.Json} {n : ℕ} {p : Spot.Program}
    {l : Spot.Slots.Layout} (h : PopMatches desc q n)
    (hp : Spot.Program.parse ⟨desc.toArray⟩ = .ok p) (hl : Spot.Slots.layout p q = .ok l) :
    (popOf l).length = n := by
  have := congrArg List.length (popOf_units h hp hl)
  simpa only [List.length_map, List.length_range] using this

/-- **The registration's strata**: the sizes of the strata of the population, the first class of the layout the deployed
verifier evaluates on `desc` under the query `q`, in slot order; none where the parse or the layout fails. -/
def nsOf (desc : Claims.Msg) (q : Lean.Json) : List ℕ :=
  match Spot.Program.parse ⟨desc.toArray⟩ with
  | .error _ => []
  | .ok p =>
    match Spot.Slots.layout p q with
    | .error _ => []
    | .ok l => (l.classes.toList.headD #[]).toList.map Array.size

theorem nsOf_eq {desc : Claims.Msg} {q : Lean.Json} {p : Spot.Program} {l : Spot.Slots.Layout}
    (hp : Spot.Program.parse ⟨desc.toArray⟩ = .ok p) (hl : Spot.Slots.layout p q = .ok l) :
    nsOf desc q = (l.classes.toList.headD #[]).toList.map Array.size := by
  unfold nsOf
  rw [hp]
  show (match Spot.Slots.layout p q with
    | .error _ => []
    | .ok l => (l.classes.toList.headD #[]).toList.map Array.size) = _
  rw [hl]

/-- A layout's strata sizes sum to its population's slots. -/
theorem sum_strata_popOf (l : Spot.Slots.Layout) :
    ((l.classes.toList.headD #[]).toList.map Array.size).sum = (popOf l).length := by
  simp only [popOf, List.length_flatten, List.map_map]
  congr 1

/-- **The strata sum to `n`**: at `PopMatches`, the registration's strata cover the population's `n` slots. -/
theorem nsOf_sum {desc : Claims.Msg} {q : Lean.Json} {n : ℕ} (h : PopMatches desc q n) : (nsOf desc q).sum = n := by
  obtain ⟨p, l, hp, hl, -⟩ := popMatches_iff.1 h
  rw [nsOf_eq hp hl, sum_strata_popOf, popOf_size h hp hl]

/-- **Slot `u` is unit `u`**: at `PopMatches`, the population's proof block at slot `u` is the population's unit `u`. -/
theorem popOf_unit {desc : Claims.Msg} {q : Lean.Json} {n : ℕ} {p : Spot.Program}
    {l : Spot.Slots.Layout} (h : PopMatches desc q n)
    (hp : Spot.Program.parse ⟨desc.toArray⟩ = .ok p) (hl : Spot.Slots.layout p q = .ok l)
    (u : ℕ) (hu : u < (popOf l).length) : ((popOf l)[u]'hu).unit = some u := by
  have hu' : u < ((popOf l).map (·.unit)).length := by rw [List.length_map]; exact hu
  have := List.getElem_of_eq (popOf_units h hp hl) hu'
  simpa only [List.getElem_map, List.getElem_range] using this

/-- **The population's slot count** of a claims partition: the proof blocks of its first class. -/
def popCount (Pt : Claims.Sampling.Partition) : ℕ := (Pt.classes.headD []).flatten.length

/-- The claims' partition of a layout has the layout's population as its population. -/
theorem popCount_slotPartition (l : Spot.Slots.Layout) : popCount (slotPartition l) = (popOf l).length := by
  rcases l with ⟨⟨cl⟩⟩
  cases cl with
  | nil => rfl
  | cons c cs => simp [popCount, slotPartition, popOf, List.length_flatten, Function.comp_def]

/-- `partOf P κ ns`'s population has one slot per unit, at strata that sum to `n`. -/
theorem popCount_partOf_eq {C : Verity.Boolean.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n)
    (κ : Fin C.N → Option Bool) {ns : List ℕ} (hns : ns.sum = n) : popCount (partOf P κ ns) = n := by
  show (ns.splitLengths ((List.finRange n).map (blockOf P κ))).flatten.length = n
  rw [List.flatten_splitLengths ((List.finRange n).map (blockOf P κ)) ns (by simp [hns]), List.length_map,
    List.length_finRange]

/-- **N = n**: at `PopMatches`, the claims' partition of the deployed layout has as many population slots as the
game's partition `partOf P κ ns` has population slots, at strata that sum to `n`. -/
theorem popCount_partOf {desc : Claims.Msg} {q : Lean.Json} {n : ℕ} {p : Spot.Program}
    {l : Spot.Slots.Layout} (h : PopMatches desc q n)
    (hp : Spot.Program.parse ⟨desc.toArray⟩ = .ok p) (hl : Spot.Slots.layout p q = .ok l)
    {C : Verity.Boolean.Circuit} (P : FlockSoundness.Audit.Partition C n) (κ : Fin C.N → Option Bool)
    {ns : List ℕ} (hns : ns.sum = n) :
    popCount (slotPartition l) = popCount (partOf P κ ns) := by
  rw [popCount_slotPartition, popOf_size h hp hl, popCount_partOf_eq P κ hns]

/-- **N = n at the registration**: where `RegMatches` names the statement's registration, and `PopMatches` holds at
the statement's descriptor and that registration's query, the statement's partition has as many population slots as
`partOf P κ ns` has population slots, at strata that sum to `n`. -/
theorem popCount_regMatches {c : SamplingDeploy.Code} {reg : Claims.Msg → Option SamplingDeploy.Reg}
    (hreg : SamplingDeploy.RegMatches c reg) {x : Claims.Msg} {s : Claims.Sampling.AuditStatement}
    {Pt : Claims.Sampling.Partition} (hx : c.parse x = some s) (hP : c.partition s = some Pt) {n : ℕ}
    (hpop : ∀ g, reg x = some g → PopMatches s.desc g.query n)
    {C : Verity.Boolean.Circuit} (P : FlockSoundness.Audit.Partition C n) (κ : Fin C.N → Option Bool)
    {ns : List ℕ} (hns : ns.sum = n) :
    popCount Pt = popCount (partOf P κ ns) := by
  obtain ⟨g, l, hg, hd, hl, hPt, -⟩ := hreg x s Pt hx hP
  rw [hPt]
  exact popCount_partOf (hpop g hg) hd hl P κ hns

end Flock.Bridge
