module

public import Definitions.Pouw.H1Zy.Chain
public import Definitions.Pouw.H1Zy.LaneBits
public import Proofs.Pouw.H1Zy.Lane
public import Definitions.Pouw.H1Zy.Combine
public import Proofs.Pouw.H1Zy.Combine
public import Proofs.Pouw.TS.Session.Group.Layer
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
import Proofs.Pouw.TS.GroupSound
import Std.Tactic.BVDecide.Normalize.Bool
import Std.Tactic.BVDecide.Normalize.Prop

/-!
# A leaf's 33 linked sessions realize one unit (the proof)

`ResourceAccounting.SecurityProofs.TS.LeafLanesSound` is the group argument (`ResourceAccounting.TS.Group.group_soundAt`) at option A's
33 statements (`Option (Fin 32)`) and their layer `leafLayer`, where `ResourceAccounting.Assumptions.TS.GroupRealizes` is proved
(`ResourceAccounting.TS.Lanes.groupRealizes_lanes32`) rather than assumed. The lanes cover the leaf
(`ResourceAccounting.TS.Lanes.cover`: cell `(i, j)` is lane `4 (i mod 8) + ⌊(j mod 8)/2⌋`'s at row slot `⌊i/8⌋` and column slot
`2⌊j/8⌋ + (j mod 2)`), so a leaf whose 33 statements each hold at their committed rows is served right
(`unitServed_of`): `LeafChecks`, which the circuits computing their Definitions give (`leafChecks_of_computes`), gives
each lane's `LaneOk` at a `K` the lanes bind at, which the leaf verifier's tile check gives (`k_of_check`), and the
combine's `CombineOk`, and each lane's view agrees with the leaf on what `LaneOk` reads (`LaneAgrees`). The leaf is the
one the lanes' committed rows decode to and the receipt the combine's (`ResourceAccounting.TS.Lanes.leafOf`, `receiptOf`): each
A row as lane `4 (r mod 8)` decodes it, each B row as lane `⌊(j mod 8)/2⌋` does, each cell of the record as the one
lane whose cell it is. The four lanes that read one A row (or one B row) read one registered tree at one row in full, so
wherever the sessions' registrants' rows agree they decode it alike (`read_bits`, `decodes_of_checks`); the known
ports' fields follow from the known-read check (`laneKnownHolds_of_check`, `combKnownOk_of_check`); the partials record
is the combine's decoded one: lane `l`'s `partials` and the combine's `p<l>` are `ownOf`'s ports of one table at the
leaf's row `l`, placed from one base row for the unit (`place`), read in full (`partials_bits`). Each statement's known
content (`knownOf`) is the `okKnown` form the group layer reads, its other known rows then its `o0` row
(`knownOf_okKnown`). The union bound over the 33 sessions gives `33 δ`.

This module imports `Proofs.Flock.SoundAtBridge` (through `Proofs.Pouw.TS.GroupSound`), whose imports make `X`
a token, so `Proofs/ResourceAccounting.lean` imports it.
-/

@[expose] public section

namespace ResourceAccounting.TS.Lanes

open Flock.SoundAtBridge

end ResourceAccounting.TS.Lanes

open ResourceAccounting.TS ResourceAccounting.TS.Lanes Flock.SoundAtBridge PouwWindow.Lanes

/-- **Option A's known content is the group layer's form**: statement `k`'s known rows at the public inputs `P`
(`knownOf`, `PouwWindow.Lanes.knownRows`) are its other known rows (lane `l`'s `admit` and `pub`, the combine's none)
followed by its `o0` row 1 under its salt (`ResourceAccounting.TS.Group.okKnown`). -/
theorem ResourceAccounting.TS.Lanes.knownOf_okKnown (P : LanePub) (salts : String → ByteArray) (k : Option (Fin 32)) :
    ∃ pre, knownOf P salts k = ResourceAccounting.TS.Group.okKnown pre (salts "o0") := by
  cases k with
  | none => exact ⟨#[], rfl⟩
  | some l =>
    exact ⟨#[("admit", #[(PouwWindow.Lanes.admitRow (List.ofFn P.admit), salts "admit")]),
      ("pub", #[(PouwWindow.Lanes.pubRow P.salt P.position P.roots l.val, salts "pub")])], rfl⟩

