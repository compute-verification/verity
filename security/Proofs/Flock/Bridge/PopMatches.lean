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

/-! `Bridge.PopMatches`, a decidable check that the registration's population is the game's units in order; it does not
state that slot `u`'s gates are unit `u`'s lowered gates. -/

@[expose] public section

namespace Flock.Bridge

open Properties.Spot

/-- **A layout's population**: its first class's proof blocks, in slot order. -/
def popOf (l : Spot.Slots.SlotOrder) : List Spot.Slots.ProofBlock :=
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

theorem popOf_units {desc : Claims.Msg} {q : Lean.Json} {n : ℕ} {p : Spot.Program}
    {l : Spot.Slots.SlotOrder} (h : PopMatches desc q n)
    (hp : Spot.Program.parse ⟨desc.toArray⟩ = .ok p) (hl : Spot.Slots.layout p q = .ok l) :
    (popOf l).map (·.unit) = (List.range n).map some := by
  obtain ⟨p', l', hp', hl', hu⟩ := popMatches_iff.1 h
  rw [hp] at hp'
  cases hp'
  rw [hl] at hl'
  cases hl'
  exact hu

theorem popOf_size {desc : Claims.Msg} {q : Lean.Json} {n : ℕ} {p : Spot.Program}
    {l : Spot.Slots.SlotOrder} (h : PopMatches desc q n)
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

theorem nsOf_eq {desc : Claims.Msg} {q : Lean.Json} {p : Spot.Program} {l : Spot.Slots.SlotOrder}
    (hp : Spot.Program.parse ⟨desc.toArray⟩ = .ok p) (hl : Spot.Slots.layout p q = .ok l) :
    nsOf desc q = (l.classes.toList.headD #[]).toList.map Array.size := by
  unfold nsOf
  rw [hp]
  show (match Spot.Slots.layout p q with
    | .error _ => []
    | .ok l => (l.classes.toList.headD #[]).toList.map Array.size) = _
  rw [hl]

theorem sum_strata_popOf (l : Spot.Slots.SlotOrder) :
    ((l.classes.toList.headD #[]).toList.map Array.size).sum = (popOf l).length := by
  simp only [popOf, List.length_flatten, List.map_map]
  congr 1

theorem nsOf_sum {desc : Claims.Msg} {q : Lean.Json} {n : ℕ} (h : PopMatches desc q n) : (nsOf desc q).sum = n := by
  obtain ⟨p, l, hp, hl, -⟩ := popMatches_iff.1 h
  rw [nsOf_eq hp hl, sum_strata_popOf, popOf_size h hp hl]

theorem popOf_unit {desc : Claims.Msg} {q : Lean.Json} {n : ℕ} {p : Spot.Program}
    {l : Spot.Slots.SlotOrder} (h : PopMatches desc q n)
    (hp : Spot.Program.parse ⟨desc.toArray⟩ = .ok p) (hl : Spot.Slots.layout p q = .ok l)
    (u : ℕ) (hu : u < (popOf l).length) : ((popOf l)[u]'hu).unit = some u := by
  have hu' : u < ((popOf l).map (·.unit)).length := by rw [List.length_map]; exact hu
  have := List.getElem_of_eq (popOf_units h hp hl) hu'
  simpa only [List.getElem_map, List.getElem_range] using this

def popCount (Pt : Claims.Sampling.Partition) : ℕ := (Pt.classes.headD []).flatten.length

theorem popCount_slotPartition (l : Spot.Slots.SlotOrder) : popCount (slotPartition l) = (popOf l).length := by
  rcases l with ⟨⟨cl⟩⟩
  cases cl with
  | nil => rfl
  | cons c cs => simp [popCount, slotPartition, popOf, List.length_flatten, Function.comp_def]

theorem popCount_partOf_eq {C : Verity.Boolean.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n)
    (κ : Fin C.N → Option Bool) {ns : List ℕ} (hns : ns.sum = n) : popCount (partOf P κ ns) = n := by
  show (ns.splitLengths ((List.finRange n).map (blockOf P κ))).flatten.length = n
  rw [List.flatten_splitLengths ((List.finRange n).map (blockOf P κ)) ns (by simp [hns]), List.length_map,
    List.length_finRange]

theorem popCount_partOf {desc : Claims.Msg} {q : Lean.Json} {n : ℕ} {p : Spot.Program}
    {l : Spot.Slots.SlotOrder} (h : PopMatches desc q n)
    (hp : Spot.Program.parse ⟨desc.toArray⟩ = .ok p) (hl : Spot.Slots.layout p q = .ok l)
    {C : Verity.Boolean.Circuit} (P : FlockSoundness.Audit.Partition C n) (κ : Fin C.N → Option Bool)
    {ns : List ℕ} (hns : ns.sum = n) :
    popCount (slotPartition l) = popCount (partOf P κ ns) := by
  rw [popCount_slotPartition, popOf_size h hp hl, popCount_partOf_eq P κ hns]

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
