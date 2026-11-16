module

public import Proofs.Flock.SpotFlock.AddressesWired

/-!
# The five facts of the address premise, at the wired program

`AddressesWired h ω keep` asks for one address map under which, at every accepted run's checked unit and every gate
its proof block needs side by side, the address holds (`CheckedRun.AddressHoldsWired`). This file states what an
address must hold as five facts, each about one address map `addr` and nothing else, so each is proved on its own;
`AddressFactsGive` proves that the five at one map give the property, and at the code's own wiring the premise
`AddressesConnected`.

Which way an address is read is not a choice. It is read as a registered value exactly where a registered read is
named like the address's port in the unit's session (`regNamed`), and off the interior record where none is. So a fact
that speaks of the read states its part of both ways, each under its case of `regNamed`.

* **The width** (`WidthFact`): the address's place is as wide as the row of its port in the unit's session
  (`RowAt`'s first fact).
* **The copy** (`CopyFact`): the bit that row holds at the place, at the unit's instance, is the one the layout
  reads for the gate (`RowAt`'s last fact).
* **The row** (`RowFact`): the unit's instance is one of its session's (`Counted`); a session read off the interior
  record has one table; and under the roots the gate is opened under, the slot's leaf count and its leaf are the
  read's: at a registered read the session header's position for the instance's row, else the sampled proof block's
  row of the published tree.
* **The entry** (`EntryFact`): where a registered read is named like the port, the port is the first of that name;
  else it is one of the session's ports; and the slot names, at the read's schema, the entry round 2 reads the value
  off: the registered read's (`readKeyAt`), else the template's port of the interior record.
* **One tree a gate** (`OneTreeFact`): a gate the wiring moves has the place and the slot of the gate it is sent to;
  and the roots the wired program opens the gate under are those round 2 reads the value under: a registered read's
  (`readRoots`), else the replay block's, the gate being one of the replay block's that the wired program doesn't
  commit as that block sees it.

The five are the parts of `AddressHoldsWired` and of the map's own condition, with nothing added and nothing left
out: the width and the copy are `RowAt`; the row, the entry and one tree a gate are `Counted`, and
`ReadsRegisteredWired` or `ReadsInteriorWired` clause by clause.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held) (ω : ℕ → ℕ) (keep : ℕ → Bool)

namespace CheckedRun

variable (r : CheckedRun h.stage)

/-- **The width, at address `S`**: the address's place is as wide as the `16 w` bits of the address's port of the
session's first table (`RowAt`'s first fact). -/
def WidthAt (S : Address) : Prop :=
  S.place.1 = 16 * (r.sts[0]!.c.ports[S.port r.u]!).words

/-- **The copy, at address `S` and the unit's gate `i`**: the bit the port's row holds at the address's place, at the
unit's instance, is the one the layout reads for `i` (`RowAt`'s last fact). -/
def CopyAt (S : Address) (i : ℕ) : Prop :=
  _root_.Flock.InnerRows.bitAt r.p r.wF 0 (r.n h / r.sts[0]!.c.g)
      (FlockSoundness.Typed.msgCol r.sts[0]!.c (r.n h % r.sts[0]!.c.g) (S.port r.u) (S.place.2 / 8) (S.place.2 % 8)) =
    _root_.Flock.InnerRows.bitAt r.p r.wF 0 (r.β h) (r.wire h i)

/-- **The row, at address `S` and the wired program's gate `g`**: the unit's instance is one of its session's
(`Counted`). Where a registered read `reg` is named like the address's port, then under the roots the wired program
opens `g` under, at an entry of `reg`'s leaf count the slot names, the slot's leaf count is `reg`'s and its leaf
(`leafAt`) is the position the session header gives the instance's row. Where none is, the session has one table, and
over any population its public file loads, under the replay block's roots, the slot's leaf count is the published
tree's and its leaf the sampled proof block's row. -/
def RowHolds (S : Address) (g : ℕ) : Prop :=
  r.Counted h ∧
    (∀ reg, regNamed r.ins r.sts (S.port r.u) = some reg →
      ∀ e, entryAt (r.rootsWired ω keep g) S.slot = some e → e.leaves = reg.leaves →
        leavesAt (r.rootsWired ω keep g) S.slot e = reg.leaves ∧
        ∀ t positions paths, r.sts[0]!.pub.tables = some t →
          _root_.Flock.Registered.entry r.sts[0]!.pub reg = .ok (positions, paths) →
          leafAt (r.rootsWired ω keep g) S.slot e =
            positions.getD (_root_.Flock.Registered.ref t r.sts[0]!.c.ports.size (S.port r.u) (r.n h)) 0) ∧
    (regNamed r.ins r.sts (S.port r.u) = none →
      r.ins.read.I.count ≤ 1 ∧
      ∀ pop, _root_.Flock.HmOut.loadPublic r.ins.read.tags r.sts[0]!.c r.ins.read.I.publicFile = .ok pop →
        ∀ e, entryAt (iroots r.x r.seen r.ρ) S.slot = some e →
          leavesAt (iroots r.x r.seen r.ρ) S.slot e = (pubPort r.sts[0]!.c pop (S.port r.u)).leaves ∧
          leafAt (iroots r.x r.seen r.ρ) S.slot e =
            rowAt r.sts[0]!.c pop (S.port r.u) (unitOf r.ins.read.draw (r.n h)))

/-- **The entry, at address `S` and the wired program's gate `g`**. Where a registered read `reg` is named like the
address's port: the port is the first of that name, and under the roots the wired program opens `g` under the slot
names the entry round 2 reads `reg`'s value off (`readKeyAt`), at `reg`'s schema. Where none is: the port is one of
the session's first table's, under the replay block's roots the slot names the template's port of the interior
record, and over any population the session's public file loads the slot has the published tree's schema. -/
def EntryHolds (S : Address) (g : ℕ) : Prop :=
  (∀ reg, regNamed r.ins r.sts (S.port r.u) = some reg →
      (List.range r.sts[0]!.c.ports.size).find? (fun q => r.sts[0]!.c.ports[q]!.name == reg.name) =
          some (S.port r.u) ∧
        entryAt (r.rootsWired ω keep g) S.slot =
          keyAt (r.rootsWired ω keep g)
            (readKeyAt h r.x r.m₁ r.c₀ r.c₁ (templateOf h.ms r.u) reg.name reg.value) ∧
        S.slot.schema = reg.schema) ∧
    (regNamed r.ins r.sts (S.port r.u) = none →
      S.port r.u < r.sts[0]!.c.ports.size ∧
        entryAt (iroots r.x r.seen r.ρ) S.slot =
          keyAt (iroots r.x r.seen r.ρ)
            (.interior (h.names.getD (templateOf h.ms r.u) "") (r.sts[0]!.c.ports[S.port r.u]!).name) ∧
        ∀ pop, _root_.Flock.HmOut.loadPublic r.ins.read.tags r.sts[0]!.c r.ins.read.I.publicFile = .ok pop →
          S.slot.schema = (pubPort r.sts[0]!.c pop (S.port r.u)).schema)

/-- **The roots, at address `S` and the wired program's gate `g`**: where a registered read `reg` is named like the
address's port, the roots the wired program opens `g` under are those round 2 reads `reg`'s value under
(`readRoots`); where none is, `g` is a gate of the unit's replay block that `ω` leaves alone and the wired program
doesn't commit as that block sees it, so the wired program opens it under the replay block's roots
(`rootsWired`). -/
def RootsHold (S : Address) (g : ℕ) : Prop :=
  (∀ reg, regNamed r.ins r.sts (S.port r.u) = some reg →
      r.rootsWired ω keep g = readRoots h r.x r.m₁ r.c₀ r.c₁ r.ρ (templateOf h.ms r.u) reg.name reg.value) ∧
    (regNamed r.ins r.sts (S.port r.u) = none →
      g ∈ cutBlock ω r.rb ∧ (r.s.prog.wired ω keep).Boundary (cutBlock ω r.rb) g = false)

end CheckedRun

variable (addr : ℕ → Address)

/-- **The width**: at every accepted run's checked unit, at each gate `i` its proof block needs side by side, the
place of `i`'s address is as wide as the row of its port in the unit's session (`CheckedRun.WidthAt`). -/
def WidthFact : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → ∀ i, r.s.prog.Needs r.b i → r.WidthAt h (addr i)

/-- **The copy**: at every accepted run's checked unit, at each gate `i` its proof block needs side by side, the bit
the row holds at the place of `i`'s address is the one the layout reads for `i` (`CheckedRun.CopyAt`). -/
def CopyFact : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → ∀ i, r.s.prog.Needs r.b i → r.CopyAt h (addr i) i

/-- **The row**: at every accepted run's checked unit, at each gate `i` its proof block needs side by side, the slot
of `i`'s address reads the unit's row, under the roots the wired program opens `ω i` under
(`CheckedRun.RowHolds`). -/
def RowFact : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → ∀ i, r.s.prog.Needs r.b i → r.RowHolds h ω keep (addr i) (ω i)

/-- **The entry**: at every accepted run's checked unit, at each gate `i` its proof block needs side by side, the
slot of `i`'s address names the entry round 2 reads the value off, under the roots the wired program opens `ω i`
under (`CheckedRun.EntryHolds`). -/
def EntryFact : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → ∀ i, r.s.prog.Needs r.b i → r.EntryHolds h ω keep (addr i) (ω i)

/-- **One tree a gate**: a gate the wiring moves has the place and the slot of the gate it is sent to, the two being
one committed bit; and at every accepted run's checked unit, at each gate `i` its proof block needs side by side, the
roots the wired program opens `ω i` under are those round 2 reads the value under (`CheckedRun.RootsHold`). So the
unit that writes a gate and each unit that reads it open it at one slot under one root. -/
def OneTreeFact : Prop :=
  (∀ i, (addr i).place = (addr (ω i)).place ∧ (addr i).slot = (addr (ω i)).slot) ∧
    ∀ r : CheckedRun h.stage, r.Ok h → ∀ i, r.s.prog.Needs r.b i → r.RootsHold h ω keep (addr i) (ω i)

end SpotFlock
