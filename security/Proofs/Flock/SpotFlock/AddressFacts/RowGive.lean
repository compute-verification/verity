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

* no proof block always checked;
* a registered read at a needed gate is a link's or a wire's (`NeededKinds`): the reference reads' ports that hold a
  commitment position are ports a link or a wire names (`refPortsNamed`, with `sourcesNamed` at the inputs), so round
  2 reads none of them at its fallback;
* the address map's slots of the two shapes `addressOf` gives them (`ShapeFact`): at a linked port the boundary port's
  row the unit reads, unsized (`linkShape`); at a wired port and at a port the session reads its own rows at, the
  writer's or the unit's place in its cell on a stride of its template's units a replay block, sized (`wireShape`,
  `ownShape`);
* a needed gate no registered read is named at is an output row (`OutputsAt`): Flock's `HmOut.loadPublic` holds an
  output row's refs to the instance's own row, and no check holds an input row's that no registered read names;
* the roots half of one tree a gate (`OneTreeFact`'s second conjunct): at a registered read the wired program's roots
  are those round 2 reads the value under (`readRoots`).
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Lean (Json)

namespace AddressRow

variable (h : Held) (ω : ℕ → ℕ) (keep : ℕ → Bool)

/-- **The reference reads name no unlinked, unwired output position**, decided on the auditor's copies: each written
bit's port (`writtenBits`) that its member's reference registered reads have an entry for is a port a link or a wire
names (`heldLink`, `heldWire`). With `sourcesNamed` at the source bits, the ports of the reference reads that hold a
commitment position are ports a link or a wire names, the direction of the design's condition the row fact reads. -/
def refPortsNamed : Bool :=
  (writtenBits h.ms).all fun s =>
    (match refEntry h s.2.1 s.2.2.row with
      | .null => true
      | _ => false) ||
      (heldLink h (h.names.getD s.2.1 "") s.2.2.row).isSome ||
      (heldWire h (h.names.getD s.2.1 "") s.2.2.row).isSome

/-- **The shape of unit `u`'s slot at a port its session reads its own rows at**: the unit's place in its template's
cell on a stride of the template's units a replay block, so unit `i·n_t + j` of the published tree at the replay
block's place `i`. -/
def ownShape (u : ℕ) : RowShape :=
  ⟨cellPos h u, h.reg.ns.getD (templateOf h.ms u) 0, 0⟩

/-- **The shape of unit `u`'s slot at a linked port**: the boundary port's row `ρ·per + x` it reads (`rowShape`). -/
def linkShape (u : ℕ) (port : String) : RowShape := rowShape h (fun _ _ _ => 0) u port .boundaryLink

/-- **The shape of unit `u`'s slot at a wired port**: the row `x` it reads on a stride of `per` (`rowShape`); at
`addressOf`, the writer's own place in its cell on a stride of its template's units a replay block. -/
def wireShape (u : ℕ) (port : String) : RowShape := rowShape h (fun _ _ _ => 0) u port .wired

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

variable (addr : ℕ → Address)

/-- **A registered read at a needed gate is a link's or a wire's**: at every accepted run's checked unit and every gate
`i` its proof block needs, a registered read named like the address's port is read, as the auditor's copies tell it
(`kindHeld`), at a link to the boundary registration or at a wire. It follows from `refPortsNamed`, `sourcesNamed`,
`NeededBits` and the run's interior registration registering no value. -/
def NeededKinds : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → ∀ i, r.s.prog.Needs r.b i → ∀ reg,
    regNamed r.ins r.sts ((addr i).port r.u) = some reg →
      kindHeld h (templateOf h.ms r.u) reg.name reg.value = .boundaryLink ∨
        kindHeld h (templateOf h.ms r.u) reg.name reg.value = .wired

/-- **The address map's slots have the shapes round 2 holds rows to**: at every accepted run's checked unit and every
gate `i` its proof block needs, where a registered read `reg` is named like the address's port, the slot fits
`linkShape` at a link and `wireShape`, sized, at a wire; where none is, it fits the unit's own (`ownShape`), sized. -/
def ShapeFact : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → ∀ i, r.s.prog.Needs r.b i →
    (∀ reg, regNamed r.ins r.sts ((addr i).port r.u) = some reg →
      (kindHeld h (templateOf h.ms r.u) reg.name reg.value = .boundaryLink →
          (linkShape h r.u reg.name).Fits (addr i).slot) ∧
        (kindHeld h (templateOf h.ms r.u) reg.name reg.value = .wired →
          (wireShape h r.u reg.name).FitsSized (addr i).slot)) ∧
    (regNamed r.ins r.sts ((addr i).port r.u) = none → (ownShape h r.u).FitsSized (addr i).slot)

/-- **A needed gate no registered read is named at is an output row**: at every accepted run's checked unit and every
gate `i` its proof block needs, where no registered read is named like the address's port, the port is an output row
of the session's circuit, whose refs Flock's `HmOut.loadPublic` holds to the instance's own row. -/
def OutputsAt : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → ∀ i, r.s.prog.Needs r.b i →
    regNamed r.ins r.sts ((addr i).port r.u) = none → _root_.Flock.HmOut.nIn r.sts[0]!.c ≤ (addr i).port r.u

/-- **What gives the row fact**: with no proof block always checked, a needed gate's registered reads a link's or a
wire's (`NeededKinds`), the address map's slots of round 2's shapes (`ShapeFact`), a needed gate read unregistered at
an output row (`OutputsAt`), and the roots the wired program opens each gate under round 2's (`RootsHold`, the second
conjunct of `OneTreeFact`), the row fact holds at the address map. -/
def RowGoal : Prop :=
  (∀ v, h.reg.always v = false) → NeededKinds h addr → ShapeFact h addr → OutputsAt h addr →
    (∀ r : CheckedRun h.stage, r.Ok h → ∀ i, r.s.prog.Needs r.b i → r.RootsHold h ω keep (addr i) (ω i)) →
    SpotFlock.RowFact h ω keep addr

end AddressRow

end SpotFlock
