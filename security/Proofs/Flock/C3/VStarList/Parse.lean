module

public import Flock.HmRow
public import Proofs.Flock.Recursive.VStarCheck
public import Properties.Flock.Circuit.Rows
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.MeasureTheory.Integral.Bochner.Basic
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
import Proofs.Flock.Soundness.ExecSetup
import VCVio.Prelude.Core

/-!
# Piece (h): V*'s files parse

What `vstarParsesScoped_of_list` rests on: `built_of_parse`, a parsed circuit's unit is a net `Net.ofRows` built.
-/

@[expose] public section

namespace FlockC3.VStarList

/-- **A parsed circuit's unit is built**: `HmRow.parse`'s nets are `Net.ofRows`'s or lookup slots, and its check
refuses a lookup slot as the unit. -/
theorem built_of_parse {tags : Flock.Tags} {cf : ByteArray} {c : Flock.Circuit}
    (h : Flock.HmRow.parse tags cf [] = .ok c) : FlockSoundness.Built c.unit := by
  have P := FlockSoundness.parse_facts h
  obtain ⟨ls, hls⟩ := P.check
  exact FlockSoundness.unit_built P (FlockSoundness.check_facts P.typed hls)

end FlockC3.VStarList
