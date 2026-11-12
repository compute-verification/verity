module

public import Proofs.Flock.SpotFlock.AddressFacts.Row
public import Proofs.Flock.SpotFlock.AddressFacts

/-!
# What the row fact of the address premise rests on

`SpotFlock.RowFact h ω keep addr` (`AddressFacts`) asks, at every accepted run's checked unit and every gate `i` its
proof block needs, that the unit's instance is one of its session's and that the slot of `i`'s address opens the
unit's row: at a registered read, under the roots the wired program opens `ω i` under, the position the session
header gives the instance's row; else, under the replay block's roots, the sampled proof block's row of the published
tree. `RowGoal` states what gives it:

* no proof block always checked, and the reference rows of registered values fixed for each place in a cell
  (`RefRows`), as `AddressRow.RowFact` takes them;
* the address map's slots of the shape round 2 holds each read's rows to (`ShapeFact`): `rowShape` at a registered
  read, by the read's kind as the auditor's copies tell it (`kindHeld`), and the unit's place in its cell on a stride
  of its template's units a replay block (`ownShape`), sized, where the port reads the session's own rows;
* a port the session reads its own rows at is an output row, or its public file has no tables (`CoverFact`): Flock's
  `HmOut.loadPublic` holds an output row's refs to the instance's own row, and nothing holds an input row's refs where
  no registered read names it;
* the roots half of one tree a gate (`OneTreeFact`'s second conjunct): at a registered read the wired program's roots
  are those round 2 reads the value under (`readRoots`).
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Lean (Json)

namespace AddressRow

variable (h : Held) (ω : ℕ → ℕ) (keep : ℕ → Bool)

/-- **The shape of unit `u`'s slot at a port its session reads its own rows at**: the unit's place in its template's
cell on a stride of the template's units a replay block, so unit `i·n_t + j` of the published tree at the replay
block's place `i`. -/
def ownShape (u : ℕ) : RowShape :=
  ⟨cellPos h u, h.reg.ns.getD (templateOf h.ms u) 0, 0⟩

/-- **A sized slot has a shape**: its position, stride and units are the shape's, and it is sized, a published tree's
slot of one row a unit. -/
def RowShape.FitsSized (sh : RowShape) (s : Slot) : Prop :=
  s.pos = sh.pos ∧ s.stride = sh.stride ∧ s.units = sh.units ∧ s.sized = true

/-- **An unsized slot's leaf count is its entry's.** -/
theorem leavesAt_fits {roots : List Msg} {s : Slot} {sh : RowShape} (e : Entry) (hf : sh.Fits s) :
    leavesAt roots s e = e.leaves := by
  unfold leavesAt; rw [hf.2.2.2]; rfl

/-- **A sized slot's leaf count is its units'.** -/
theorem leavesAt_sized {roots : List Msg} {s : Slot} {sh : RowShape} (e : Entry) (hf : sh.FitsSized s) :
    leavesAt roots s e = unitsAt roots s := by
  unfold leavesAt; rw [hf.2.2.2]; rfl

/-- **A sized slot of some units opens its unit**: leaf `pos + place·stride`, one leaf a unit. -/
theorem leafAt_sized {roots : List Msg} {s : Slot} {sh : RowShape} (e : Entry) (hf : sh.FitsSized s)
    (hu : 0 < unitsAt roots s) : leafAt roots s e = s.pos + placeOf roots * s.stride := by
  unfold leafAt
  split
  · omega
  · rw [leavesAt_sized e hf, Nat.div_self hu, Nat.mul_one]

variable (R : ℕ → String → ℕ → ℕ) (addr : ℕ → Address)

/-- **The address map's slots have the shapes round 2 holds rows to**: at every accepted run's checked unit and every
gate `i` its proof block needs, where a registered read `reg` is named like the address's port, the slot fits the
read's shape (`rowShape`, at the kind `kindHeld` tells); where none is, it fits the unit's own (`ownShape`), sized. -/
def ShapeFact : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → ∀ i, r.s.prog.Needs r.b i →
    (∀ reg, regNamed r.ins r.sts ((addr i).port r.u) = some reg →
      (rowShape h R r.u reg.name (kindHeld h (templateOf h.ms r.u) reg.name reg.value)).Fits (addr i).slot) ∧
    (regNamed r.ins r.sts ((addr i).port r.u) = none → (ownShape h r.u).FitsSized (addr i).slot)

/-- **A port a session reads its own rows at is an output row, or the file has no tables**: at every accepted run's
checked unit, each port no registered read is named like is an output row of the session's circuit, or the public file
its session loads has no tables. Flock holds an output row's refs to the instance's own row (`HmOut.loadPublic`), and
an input row's only where a registered or a shared read names it. -/
def CoverFact : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → ∀ q, regNamed r.ins r.sts q = none →
    ∀ pop, _root_.Flock.HmOut.loadPublic r.ins.read.tags r.sts[0]!.c r.ins.read.I.publicFile = .ok pop →
      _root_.Flock.HmOut.nIn r.sts[0]!.c ≤ q ∨ pop.tables = none

/-- **What gives the row fact**: with no proof block always checked, the reference rows fixed for each place in a cell
(`RefRows`), the address map's slots of round 2's shapes (`ShapeFact`), a session's own rows read at output rows or
without tables (`CoverFact`), and the roots the wired program opens each gate under round 2's (`RootsHold`, the second
conjunct of `OneTreeFact`), the row fact holds at the address map. -/
def RowGoal : Prop :=
  (∀ v, h.reg.always v = false) → RefRows h R → ShapeFact h R addr → CoverFact h →
    (∀ r : CheckedRun h.stage, r.Ok h → ∀ i, r.s.prog.Needs r.b i → r.RootsHold h ω keep (addr i) (ω i)) →
    SpotFlock.RowFact h ω keep addr

end AddressRow

end SpotFlock
