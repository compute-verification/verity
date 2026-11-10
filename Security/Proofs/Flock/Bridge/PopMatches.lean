module

public import Proofs.Flock.Bridge.Defs
public import Proofs.Flock.Claims.Sampling.Reference.DeployedCode
public import Proofs.Flock.Claims.Sampling.Select.Misses
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
(`Spot.Program.parse`), and the sampled-proofs run's population check passes on them under the registration's
query `q` (`Spot.Slots.popOk`, which reads the query as JSON, `Reg.query`): the population's units, stratum by
stratum (`Slots.popUnits`), are `0, …, n − 1`, each with its gates. It reads no circuit file: the claims' layout of the
circuit file (`Slots.fileLayout`) lays those units in the registration's strata, slot `u` unit `u`, and that slot `u`'s
gates are unit `u`'s proof block in the game's partition is `FilePartTarget`.

* `popMatches_iff`: `PopMatches` is that the parse succeeds and the population's units are `0, …, n − 1`, in order.
* `popUnits_ok`: the run's population is the registration's strata's units (`Partition.strata`), stratum by stratum.
* `nsOf desc q`: the sizes of the registration's strata, one per template in the order of its first unit, each its
  ranges' units: the sizes the claims' layout lays (`FilePart.sizes`, `sizes_ok`). `nsOf_eq` reads them off the strata,
  and `nsOf_sum`: at `PopMatches`, they sum to `n`.
-/

@[expose] public section

namespace Flock.Bridge

open Properties.Spot

/-- `PopMatches`'s check: the descriptor parses and the run's population check passes on it (`Slots.popOk`). -/
def popCheck (desc : Claims.Msg) (q : Lean.Json) (n : ℕ) : Bool :=
  match Spot.Program.parse ⟨desc.toArray⟩ with
  | .error _ => false
  | .ok p => Spot.Slots.popOk p q n

/-- **The registration's population is `n` units, in order**: `desc` parses, and its population under the query `q`
(`Slots.popUnits`) is the units `0, …, n − 1`, stratum by stratum, each with its gates. -/
@[reducible] def PopMatches (desc : Claims.Msg) (q : Lean.Json) (n : ℕ) : Prop := popCheck desc q n = true

instance instDecidablePopMatches (desc : Claims.Msg) (q : Lean.Json) (n : ℕ) : Decidable (PopMatches desc q n) :=
  inferInstanceAs (Decidable (_ = true))

theorem popMatches_iff {desc : Claims.Msg} {q : Lean.Json} {n : ℕ} :
    PopMatches desc q n ↔ ∃ p us, Spot.Program.parse ⟨desc.toArray⟩ = .ok p ∧
      Spot.Slots.popUnits p q = .ok us ∧ us.toList = List.range n := by
  unfold PopMatches popCheck
  cases hp : Spot.Program.parse ⟨desc.toArray⟩ with
  | error e => simp
  | ok p =>
    unfold Spot.Slots.popOk
    cases hus : Spot.Slots.popUnits p q with
    | error e => simp [hus]
    | ok us => simp [hus]

/-- **The run's population is the registration's strata's units**, stratum by stratum (`Partition.strata`). -/
theorem popUnits_ok {p : Spot.Program} {q : Lean.Json} {us : Array ℕ}
    (h : Spot.Slots.popUnits p q = .ok us) :
    ∃ S, Spot.Partition.strata p q = .ok S ∧
      us.toList = S.toList.flatMap fun s => SecurityProofs.expand s.2.toList := by
  unfold Spot.Slots.popUnits at h
  obtain ⟨rs, he, h⟩ := SecurityProofs.bind_eq_ok.1 h
  obtain ⟨gs, -, h⟩ := SecurityProofs.bind_eq_ok.1 h
  refine ⟨_, SecurityProofs.strata_eq p q rs he, ?_⟩
  dsimp only at h
  split at h
  · exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind])
  · obtain rfl := (Except.ok.inj h).symm
    rw [Array.toList_flatMap]
    refine List.flatMap_congr fun s _ => ?_
    obtain ⟨t, ranges⟩ := s
    rw [← SecurityProofs.toList_expandArr]

/-- **The registration's strata**: the sizes of the strata of the population under the query `q` (`Partition.strata`),
each its ranges' units, in slot order, as the claims' layout lays them (`Slots.fileLayout`); none where the parse or the
strata fail. -/
def nsOf (desc : Claims.Msg) (q : Lean.Json) : List ℕ :=
  match Spot.Program.parse ⟨desc.toArray⟩ with
  | .error _ => []
  | .ok p =>
    match Spot.Partition.strata p q with
    | .error _ => []
    | .ok S => S.toList.map fun s => (SecurityProofs.expand s.2.toList).length

theorem nsOf_eq {desc : Claims.Msg} {q : Lean.Json} {p : Spot.Program}
    {S : Array (String × Array (ℕ × ℕ))} (hp : Spot.Program.parse ⟨desc.toArray⟩ = .ok p)
    (hS : Spot.Partition.strata p q = .ok S) :
    nsOf desc q = S.toList.map fun s => (SecurityProofs.expand s.2.toList).length := by
  simp only [nsOf, hp, hS]

/-- **The strata sum to `n`**: at `PopMatches`, the registration's strata cover the population's `n` units. -/
theorem nsOf_sum {desc : Claims.Msg} {q : Lean.Json} {n : ℕ} (h : PopMatches desc q n) : (nsOf desc q).sum = n := by
  obtain ⟨p, us, hp, hus, hn⟩ := popMatches_iff.1 h
  obtain ⟨S, hS, hu⟩ := popUnits_ok hus
  have hlen := congrArg List.length hu
  rw [List.length_flatMap] at hlen
  rw [nsOf_eq hp hS, ← hlen, hn, List.length_range]

end Flock.Bridge
