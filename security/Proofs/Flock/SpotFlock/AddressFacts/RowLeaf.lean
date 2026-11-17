module

public import Proofs.Flock.SpotFlock.AddressFacts.RowGive

/-!
# The leaf a slot of the row fact's shapes opens, and the row round 2 holds it to

At a link the slot is unsized with no units, so under any roots it opens its position (`leafAt_link`); round 2 holds
the slot to its cell's replay block's row `b · per + x` of the boundary port (`linkedRow_eq`). At a wire the slot is
sized with no units of its own, so under the replay block's interior roots it opens `x + i · per` at the replay
block's place `i` among the interior record's, of `q · per` leaves (`leafAt_wire`, `leavesAt_wire`); round 2 holds
the slot to `span · per + x` (`wiredRow_eq`).
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Lean (Json)

namespace AddressRow

variable (h : Held)

theorem linkShape_stride (u : ℕ) (port : String) : (linkShape h u port).stride = 0 := rfl

theorem linkShape_units (u : ℕ) (port : String) : (linkShape h u port).units = 0 := rfl

theorem wireShape_units (u : ℕ) (port : String) : (wireShape h u port).units = 0 := rfl

/-- **A linked slot opens its position**, under any roots. -/
theorem leafAt_link {roots : List Msg} {s : Slot} {u : ℕ} {port : String} (e : Entry)
    (hf : (linkShape h u port).Fits s) : leafAt roots s e = (linkShape h u port).pos := by
  have hu : unitsAt roots s = 0 := by
    unfold unitsAt
    rw [hf.2.2.1, hf.2.1, linkShape_units, linkShape_stride, Nat.mul_zero]
  unfold leafAt
  split
  · exact hf.1
  · contradiction

/-- **A wired slot's leaf count under the interior roots**: `q · per`, `q` the interior record's replay blocks. -/
theorem leavesAt_wire {m₁ : Msg} {rec : Json} (hm : jsonMsg m₁ = some rec) (ρ : ℕ) {s : Slot} {u : ℕ}
    {port : String} (e : Entry) (hf : (wireShape h u port).FitsSized s) :
    leavesAt (irootsOf m₁ ρ) s e = (blocksOf rec).length * (wireShape h u port).stride := by
  rw [leavesAt_sized e hf]
  unfold unitsAt
  rw [hf.2.2.1, hf.2.1, wireShape_units, qOf_irootsOf hm, Nat.zero_add]

/-- **A wired slot of some units opens its unit's row under the interior roots**: `x + i · per`, `i` the replay
block's place among the interior record's. -/
theorem leafAt_wire {m₁ : Msg} {rec : Json} (hm : jsonMsg m₁ = some rec) (ρ : ℕ) {s : Slot} {u : ℕ}
    {port : String} (e : Entry) (hf : (wireShape h u port).FitsSized s)
    (hq : 0 < (blocksOf rec).length * (wireShape h u port).stride) :
    leafAt (irootsOf m₁ ρ) s e =
      (wireShape h u port).pos + (blocksOf rec).idxOf ρ * (wireShape h u port).stride := by
  have hu : unitsAt (irootsOf m₁ ρ) s = (blocksOf rec).length * (wireShape h u port).stride := by
    unfold unitsAt
    rw [hf.2.2.1, hf.2.1, wireShape_units, qOf_irootsOf hm, Nat.zero_add]
  rw [leafAt_sized e hf (hu ▸ hq), placeOf_irootsOf hm, hf.1, hf.2.1]

/-- **Round 2's row at a link**: slot `u`'s cell's replay block `b`, its unit's row `x` among `per`. -/
theorem linkedRow_eq {own : Spot.SecondStage.Own} {p : Spot.SecondStage.Part} {r : Spot.SecondStage.Rows}
    {u b x : ℕ} (hb : own.sample[p.span u]? = some b) (hx : Spot.SecondStage.rowAt r.runs (p.unit u) = some x)
    (hlt : x < r.per) : Spot.SecondStage.linkedRow own p r u = some (b * r.per + x) := by
  simp [Spot.SecondStage.linkedRow, hb, hx, hlt]

/-- **Round 2's row at a wire**: slot `u`'s place `span` in the sample, its unit's row `x` among `per`. -/
theorem wiredRow_eq {p : Spot.SecondStage.Part} {r : Spot.SecondStage.Rows} {u x : ℕ}
    (hx : Spot.SecondStage.rowAt r.runs (p.unit u) = some x) (hlt : x < r.per) :
    Spot.SecondStage.wiredRow p r u = some (p.span u * r.per + x) := by
  simp [Spot.SecondStage.wiredRow, hx, hlt]

end AddressRow

end SpotFlock
